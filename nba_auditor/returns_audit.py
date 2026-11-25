"""`audit_returns`: statistical red-team of a strategy return series.

Conventions
-----------
* Returns are simple (not log) per-period returns in decimals (0.01 = 1%).
* Turnover is one-way traded notional / capital per period: sum over assets of |change in weight|.
  Building a full position from cash is turnover 1; flipping from +1 to -1 is turnover 2.
* Cost in bps is charged per unit of turnover: net = gross - turnover * bps / 1e4.
* Unless `returns_are_net=True`, the input returns are treated as gross of costs.
* The headline series is net at `costs_bps` (or 10 bps if turnover is known and no cost was given),
  otherwise the input series as given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .catalog import DISCLAIMER, NO_BENCHMARK_REASON, SCORE_CAVEAT
from .parsing import InputError, parse_series, periods_per_year, read_table
from .stats import (deflated_sharpe, gaussian_top_share, hac_bandwidth, lo_sharpe, max_drawdown,
                    min_track_record_length, moments, newey_west_tstat, probabilistic_sharpe, variance_ratio)

COST_GRID_BPS = (0.0, 5.0, 10.0, 20.0)
DEFAULT_COST_BPS = 10.0
DSR_PASS, DSR_FAIL = 0.95, 0.50
IMPLAUSIBLE_SHARPE, ARTIFACT_SHARPE, IMPLAUSIBLE_HIT = 3.0, 5.0, 0.65
WEIGHTS = {"dsr": 40, "costs": 20, "stability": 20, "concentration": 20}
MIN_HAC_T = 2.0

VERDICT_NO_FLAGS, VERDICT_SOME_FLAGS, VERDICT_MAJOR_FLAGS = "No red flags found", "Some red flags", "Major red flags"
VERDICT_NOTE = "No red flags doesn't mean a strategy will make money; it means these checks didn't find the common errors."


def _clip01(x: float) -> float:
    return 0.0 if x != x else max(0.0, min(1.0, x))  # NaN -> 0


def _f(x: Any, nd: int = 6) -> Any:
    """JSON-friendly rounding; NaN/inf become None/'inf'."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return round(float(x), nd)
    return x


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _f(obj)


def _label(idx: pd.Index, i: int) -> str:
    if i < 0:
        return "start"
    v = idx[i]
    return str(v.date()) if isinstance(v, pd.Timestamp) else str(v)


@dataclass
class ReturnsAudit:
    verdict: str
    reasons: list[str]
    score: int
    score_breakdown: list[dict]
    sections: dict
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    tool: str = "audit_returns"

    def to_dict(self) -> dict:
        return _clean({
            "tool": self.tool,
            "verdict": self.verdict,
            "verdict_note": VERDICT_NOTE,
            "reasons": self.reasons,
            "audit_score": self.score,
            "audit_score_note": SCORE_CAVEAT,
            "score_breakdown": self.score_breakdown,
            **self.sections,
            "warnings": self.warnings,
            "assumptions": self.assumptions,
            "disclaimer": DISCLAIMER,
        })

    def to_markdown(self) -> str:
        from .report import returns_audit_markdown

        return returns_audit_markdown(self)


# ---------------------------------------------------------------------------------------------
def turnover_from_positions(positions: Any) -> pd.Series:
    """One-way turnover per period from a weights table.

    Wide format: a date column plus one column of weights per asset. Long format: date, ticker, weight.
    """
    df = read_table(positions, "positions")
    cols = {str(c).lower(): c for c in df.columns}
    dcol = next((cols[c] for c in ("date", "datetime", "time", "timestamp") if c in cols), None)
    if dcol is None:
        raise InputError("positions: needs a 'date' column")
    tcol = next((cols[c] for c in ("ticker", "symbol", "asset") if c in cols), None)
    wcol = next((cols[c] for c in ("weight", "w", "position", "pos") if c in cols), None)
    df[dcol] = pd.to_datetime(df[dcol], format="mixed")
    if tcol is not None and wcol is not None:
        wide = df.pivot_table(index=dcol, columns=tcol, values=wcol, aggfunc="sum").fillna(0.0)
    else:
        wide = df.set_index(dcol).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    wide = wide.sort_index()
    to = wide.diff().abs().sum(axis=1)
    to.iloc[0] = wide.iloc[0].abs().sum()
    to.name = "turnover"
    return to


def _align_turnover(r: pd.Series, turnover: Any, positions: Any, warnings: list[str]) -> pd.Series | None:
    if turnover is None and positions is None:
        return None
    if positions is not None:
        to = turnover_from_positions(positions)
    elif isinstance(turnover, (int, float)) and not isinstance(turnover, bool):
        if turnover < 0:
            raise InputError("turnover must be >= 0")
        return pd.Series(float(turnover), index=r.index, name="turnover")
    else:
        to = parse_series(turnover, "turnover")
    if (to < 0).any():
        raise InputError("turnover must be >= 0 (it is sum |change in weight|)")
    if isinstance(r.index, pd.DatetimeIndex) and isinstance(to.index, pd.DatetimeIndex):
        aligned = to.reindex(r.index)
        missing = int(aligned.isna().sum())
        if missing:
            warnings.append(f"Turnover was missing on {missing} of {len(r)} return dates; those were set to 0 "
                            "(this flatters net results if trades happened then).")
        return aligned.fillna(0.0)
    if len(to) != len(r):
        raise InputError(f"turnover has {len(to)} values but returns have {len(r)}; give dates or equal lengths")
    return pd.Series(to.to_numpy(), index=r.index, name="turnover")


def _stats_block(x: np.ndarray, ppy: float) -> dict:
    mu, sd, skew, kurt = moments(x)
    sr = mu / sd if sd > 0 else float("nan")
    return {"n": len(x), "mean_ann": mu * ppy, "vol_ann": sd * math.sqrt(ppy), "sharpe_ann": sr * math.sqrt(ppy),
            "t_newey_west": newey_west_tstat(x, hac_bandwidth(x)), "skew": skew, "kurtosis": kurt, "hit_rate": float((x > 0).mean())}


def audit_returns(
    returns: Any,
    freq: str | float = "auto",
    costs_bps: float | None = None,
    turnover: Any = None,
    benchmark: Any = None,
    n_trials: int | None = None,
    positions: Any = None,
    trials_sr_variance: float | None = None,
    input_kind: str = "returns",
    returns_are_net: bool = False,
    units: str = "decimal",
    _context: dict | None = None,
) -> ReturnsAudit:
    """Audit a strategy's return series. See the module docstring for conventions.

    Args:
        returns: per-period returns (CSV/JSON text, path, list, dict or Series). With dates if possible.
        freq: 'auto' (default: inferred from the dates), 'D', 'W', 'M', 'Q', 'Y', or periods per year.
            A stated frequency that contradicts the date spacing is an error.
        costs_bps: your all-in cost per unit of turnover, in basis points (headline net uses it).
        turnover: one-way turnover per period (scalar average or a series).
        benchmark: benchmark returns, same format as `returns`.
        n_trials: how many strategy variants you tried in total before picking this one.
        positions: weights table to derive turnover from (instead of `turnover`).
        trials_sr_variance: variance of the *annualized* Sharpe ratios across your trials, if known.
            Default: 1/T per period, i.e. what zero-skill trials would produce.
        input_kind: 'returns' or 'equity' (an equity curve / NAV that will be converted to returns).
        returns_are_net: True if `returns` already include costs.
        units: 'decimal' (0.01 = 1%) or 'percent' (1 = 1%).
    """
    ctx = _context or {}
    warnings: list[str] = list(ctx.get("warnings", []))
    assumptions: list[str] = list(ctx.get("assumptions", []))

    r = parse_series(returns, "returns")
    if input_kind not in ("returns", "equity"):
        raise InputError("input_kind must be 'returns' or 'equity'")
    if input_kind == "equity":
        if (r <= 0).any():
            raise InputError("equity values must be positive")
        r = r.pct_change().iloc[1:]
    if units not in ("decimal", "percent"):
        raise InputError("units must be 'decimal' or 'percent'")
    if units == "percent":
        r = r / 100.0
    n_nan = int(r.isna().sum())
    if n_nan:
        warnings.append(f"Dropped {n_nan} missing value(s) from the return series.")
        r = r.dropna()
    if len(r) < 30:
        raise InputError(f"returns: need at least 30 periods to audit, got {len(r)}")
    blockers: list[str] = []
    if units == "decimal" and float(r.abs().median()) > 0.2:
        warnings.append("The typical absolute return is above 20% per period. If these are percentages, pass units='percent'.")
        blockers.append("Return units look wrong (typical |return| above 20% per period); check units before trusting anything.")
    if (r <= -1).any():
        warnings.append("Some returns are <= -100% (a total loss in one period). Check the units and the data.")

    ppy, freq_label = periods_per_year(freq, r.index)
    if ppy > 252 * 1.01:
        warnings.append("Intraday frequency: the implausibility check (RA09) is calibrated for daily or slower bars.")
        blockers.append("Intraday series: the audit's checks are calibrated for daily or slower bars, so it does not "
                        "issue 'No red flags found' here.")
    if "assumed" in freq_label:
        blockers.append("The frequency was assumed (no dates and freq='auto'), so every annualized number may be "
                        "mis-scaled; pass dates or an explicit freq.")
    if isinstance(r.index, pd.DatetimeIndex) and len(r) >= 30:
        gaps = np.diff(r.index.values).astype("timedelta64[s]").astype(float) / 86400
        med = float(np.median(gaps))
        share = float((gaps > 5 * max(med, 1.0)).mean())
        if share > 0.05:
            msg = (f"Irregular date spacing: {share:.0%} of gaps are more than 5x the typical gap ({med:g} days). "
                   "Mixed frequencies or missing stretches distort every annualized number.")
            warnings.append(msg)
            blockers.append(msg)
    to = _align_turnover(r, turnover, positions, warnings)

    # ---------------- costs
    costs: dict[str, Any] = {"assessed": to is not None or returns_are_net, "returns_are_net": returns_are_net}
    grid = sorted(set(COST_GRID_BPS) | ({float(costs_bps)} if costs_bps is not None else set()))
    headline_cost = 0.0
    if to is not None:
        mean_to = float(to.mean())
        costs["turnover_per_period"] = mean_to
        costs["turnover_per_year"] = mean_to * ppy
        rows = []
        for c in grid:
            net = r - to * c / 1e4
            mu, sd, *_ = moments(net.to_numpy())
            rows.append({"cost_bps": c, "mean_ann": mu * ppy, "sharpe_ann": mu / sd * math.sqrt(ppy) if sd > 0 else float("nan"),
                         "total_return": float(np.prod(1 + net.to_numpy()) - 1)})
        costs["sensitivity"] = rows
        costs["break_even_bps"] = float(r.mean() / mean_to * 1e4) if mean_to > 0 else float("inf")
        headline_cost = float(costs_bps) if costs_bps is not None else DEFAULT_COST_BPS
        if costs_bps is None:
            assumptions.append(f"No cost given: the headline result charges {DEFAULT_COST_BPS:g} bps per unit of turnover.")
        h = r - to * headline_cost / 1e4
        costs["headline_cost_bps"] = headline_cost
        if returns_are_net:
            assumptions.append("returns_are_net=True: the cost grid shows *additional* costs on top of the costs already included.")
            h = r - to * (float(costs_bps) if costs_bps is not None else 0.0) / 1e4
            costs["headline_cost_bps"] = float(costs_bps) if costs_bps is not None else 0.0
    else:
        h = r
        if returns_are_net:
            assumptions.append("returns_are_net=True: the returns are taken to include all trading costs.")
        else:
            warnings.append("No turnover or positions given, so costs could not be assessed: the headline result is "
                            "gross of trading costs.")
    headline_label = ("net" if (to is not None and costs.get("headline_cost_bps", 0) > 0) or returns_are_net else "gross")
    costs["headline_basis"] = headline_label

    x = h.to_numpy()
    T = len(x)
    perf = _stats_block(x, ppy)
    perf["gross_mean_ann"] = float(r.mean() * ppy)
    perf["total_return"] = float(np.prod(1 + x) - 1)
    perf["years"] = T / ppy
    L = hac_bandwidth(x)
    vr = max(1.0, variance_ratio(x, L))
    t_eff = T / vr
    perf["t_newey_west"] = newey_west_tstat(x, L)
    lo = lo_sharpe(x, ppy, max_lag=L)
    perf.update({"t_lo": lo["t_lo"], "t_iid": lo["t_iid"], "sharpe_ann_lo": lo["sharpe_ann_lo"],
                 "lag1_autocorrelation": lo["rho_1"], "newey_west_lags": L, "variance_ratio": vr,
                 "effective_periods": t_eff})
    if vr > 1.5:
        warnings.append(f"Returns are autocorrelated (long-run variance {vr:.1f}x the plain variance, e.g. overlapping "
                        f"holding periods or smoothing): significance uses an effective sample of {t_eff:.0f} "
                        f"instead of {T} periods.")
    if isinstance(h.index, pd.DatetimeIndex):
        perf["start"], perf["end"] = str(h.index[0].date()), str(h.index[-1].date())

    # ---------------- multiple testing
    mu, sd, skew, kurt = moments(x)
    sr = mu / sd if sd > 0 else float("nan")
    n_eff = int(n_trials) if n_trials else 1
    if n_trials is not None and n_trials < 1:
        raise InputError("n_trials must be >= 1")
    if trials_sr_variance is not None:
        if trials_sr_variance <= 0:
            raise InputError("trials_sr_variance must be > 0")
        v_sr = trials_sr_variance / ppy
        v_note = f"variance of annualized Sharpe across trials = {trials_sr_variance:g} (user supplied)"
        floor_ann = ppy / t_eff
        if trials_sr_variance < floor_ann:
            msg = (f"trials_sr_variance={trials_sr_variance:g} is below {floor_ann:.3g}, the variance that sampling "
                   "noise alone gives to independent trials of this length. That shrinks the multiple-testing hurdle; "
                   "it is only right if your trials were nearly identical (then n_trials should count distinct ideas).")
            warnings.append(msg)
            blockers.append("trials_sr_variance is below the zero-skill sampling variance, so the Deflated Sharpe Ratio "
                            "hurdle is understated.")
    else:
        v_sr = 1.0 / t_eff
        v_note = ("variance of Sharpe across trials assumed 1/T_eff (what zero-skill trials of this length produce; "
                  "T_eff = T / long-run variance ratio)")
    # DSR/PSR/MinTRL assume independent returns; autocorrelation is handled with an effective sample size.
    dsr = deflated_sharpe(sr, t_eff, skew, kurt, n_eff, v_sr)
    psr0 = probabilistic_sharpe(sr, 0.0, t_eff, skew, kurt)
    trl0 = min_track_record_length(sr, 0.0, skew, kurt) * vr
    trl_d = min_track_record_length(sr, dsr["sr0"], skew, kurt) * vr
    mt = {
        "n_trials": n_eff, "n_trials_declared": n_trials is not None,
        "expected_max_sharpe_ann": dsr["sr0"] * math.sqrt(ppy), "deflated_sharpe_ratio": dsr["dsr"],
        "probabilistic_sharpe_vs_0": psr0, "sr_variance_note": v_note,
        "min_track_record_periods_vs_0": trl0, "min_track_record_periods_vs_hurdle": trl_d,
        "min_track_record_years_vs_hurdle": trl_d / ppy if math.isfinite(trl_d) else float("inf"),
        "track_record_periods": T, "effective_periods": t_eff,
    }
    if n_trials is None:
        warnings.append("n_trials not given: the Deflated Sharpe Ratio assumes this is the only variant you tried. "
                        "If you tried others, the true hurdle is higher.")

    # ---------------- stability
    thirds = []
    for part in np.array_split(np.arange(T), 3):
        seg = x[part]
        st = _stats_block(seg, ppy)
        st.update({"start": _label(h.index, int(part[0])), "end": _label(h.index, int(part[-1]))})
        thirds.append(st)
    years = []
    if isinstance(h.index, pd.DatetimeIndex):
        for yr, seg in h.groupby(h.index.year):
            v = seg.to_numpy()
            if len(v) < max(2, ppy / 12):
                continue
            mu_y, sd_y, *_ = moments(v) if len(v) > 1 else (v.mean(), float("nan"), 0, 3)
            years.append({"year": int(yr), "n": len(v), "return": float(np.prod(1 + v) - 1),
                          "sharpe_ann": mu_y / sd_y * math.sqrt(ppy) if sd_y and sd_y > 0 else float("nan")})
    frac_thirds = sum(t["mean_ann"] > 0 for t in thirds) / 3
    frac_years = (sum(y["return"] > 0 for y in years) / len(years)) if len(years) >= 2 else None
    stability = {"thirds": thirds, "years": years, "positive_thirds": int(round(frac_thirds * 3)),
                 "positive_years_share": frac_years}

    # ---------------- concentration
    k = max(1, int(math.ceil(0.05 * T)))
    top = np.sort(x)[::-1][:k]
    total = float(x.sum())
    conc = {"top_periods": k, "top_5pct_sum": float(top.sum()), "total_sum": total,
            "sum_without_top_5pct": total - float(top.sum())}
    if total > 0:
        conc["top_5pct_share_of_pnl"] = float(top.sum() / total)
        conc["gaussian_expected_share"] = gaussian_top_share(mu, sd, T)
        ratio5 = conc["top_5pct_share_of_pnl"] / conc["gaussian_expected_share"]
        k1 = max(1, int(math.ceil(0.01 * T)))
        conc["top_1pct_share_of_pnl"] = float(np.sort(x)[::-1][:k1].sum() / total)
        conc["gaussian_expected_share_1pct"] = gaussian_top_share(mu, sd, T, 0.01)
        ratio1 = conc["top_1pct_share_of_pnl"] / conc["gaussian_expected_share_1pct"]
        conc["ratio_top_5pct"], conc["ratio_top_1pct"] = ratio5, ratio1
        conc["concentration_ratio"] = max(ratio5, ratio1)
    else:
        conc["top_5pct_share_of_pnl"] = float("nan")
        conc["concentration_ratio"] = float("nan")

    # ---------------- drawdown
    dd = max_drawdown(x)
    drawdown = {"max_drawdown": dd["max_drawdown"], "peak": _label(h.index, dd["peak_index"]),
                "trough": _label(h.index, dd["trough_index"]),
                "longest_underwater_periods": dd["longest_underwater_periods"],
                "longest_underwater_years": dd["longest_underwater_periods"] / ppy,
                "longest_underwater_end": _label(h.index, dd["longest_underwater_end_index"]),
                "currently_underwater": dd["currently_underwater"]}

    # ---------------- benchmark
    bench = None
    if benchmark is not None:
        b = parse_series(benchmark, "benchmark")
        if units == "percent":
            b = b / 100.0
        if isinstance(h.index, pd.DatetimeIndex) and isinstance(b.index, pd.DatetimeIndex):
            j = pd.concat([h.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
        elif len(b) == len(h):
            j = pd.DataFrame({"s": x, "b": b.to_numpy()}).dropna()
        else:
            raise InputError("benchmark: needs dates that overlap the returns, or the same length")
        if len(j) < 30:
            raise InputError(f"benchmark: only {len(j)} overlapping periods")
        if len(j) < 0.9 * T:
            warnings.append(f"Benchmark overlaps only {len(j)} of {T} periods; benchmark statistics use the overlap.")
        ex = (j["s"] - j["b"]).to_numpy()
        bm = j["b"].to_numpy()
        var_b = float(np.var(bm, ddof=1))
        beta = float(np.cov(j["s"], bm, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
        alpha = (float(j["s"].mean()) - beta * float(bm.mean())) * ppy
        resid = j["s"].to_numpy() - beta * bm
        scale = max(float(np.std(j["s"].to_numpy())), 1e-300)
        if float(np.std(resid)) < 1e-9 * scale:  # exact linear fit (e.g. leveraged benchmark): residual is rounding noise
            t_alpha = 0.0 if abs(float(resid.mean())) < 1e-9 * scale else math.copysign(float("inf"), float(resid.mean()))
        else:
            t_alpha = newey_west_tstat(resid, hac_bandwidth(resid))
        bench = {"n_overlap": len(j), "excess_mean_ann": float(ex.mean() * ppy),
                 "tracking_error_ann": float(ex.std(ddof=1) * math.sqrt(ppy)),
                 "information_ratio": float(ex.mean() / ex.std(ddof=1) * math.sqrt(ppy)) if ex.std(ddof=1) > 0 else float("nan"),
                 "t_newey_west_excess": newey_west_tstat(ex, hac_bandwidth(ex)), "beta": beta, "alpha_ann": alpha,
                 "t_newey_west_alpha": t_alpha,
                 "correlation": float(np.corrcoef(j["s"], bm)[0, 1]),
                 "benchmark_sharpe_ann": float(bm.mean() / bm.std(ddof=1) * math.sqrt(ppy)) if bm.std(ddof=1) > 0 else float("nan"),
                 "benchmark_total_return": float(np.prod(1 + bm) - 1),
                 "strategy_total_return_overlap": float(np.prod(1 + j["s"].to_numpy()) - 1)}

    # ---------------- implausibility
    implaus = []
    if ppy <= 252 * 1.01:
        if perf["sharpe_ann"] > IMPLAUSIBLE_SHARPE:
            implaus.append(f"annualized Sharpe of {perf['sharpe_ann']:.1f}")
        if T >= 250 and ppy >= 250 and perf["hit_rate"] > IMPLAUSIBLE_HIT:
            implaus.append(f"daily hit rate of {perf['hit_rate']:.0%}")

    # ---------------- score
    comps = []

    def comp(key: str, title: str, frac: float | None, detail: str) -> None:
        w = WEIGHTS[key]
        comps.append({"check": key, "title": title, "weight": w, "assessed": frac is not None,
                      "points": round(w * frac, 1) if frac is not None else None, "detail": detail})

    comp("dsr", "Deflated Sharpe Ratio (RA03)", _clip01((dsr["dsr"] - DSR_FAIL) / (DSR_PASS - DSR_FAIL)),
         f"DSR {dsr['dsr']:.2f} with n_trials={n_eff}; full points at >= {DSR_PASS}, zero at <= {DSR_FAIL}")
    if to is not None:
        be = costs["break_even_bps"]
        comp("costs", "Cost robustness (RA01)", _clip01((be - 5) / 20) if math.isfinite(be) else 1.0,
             f"break-even {be:.1f} bps; full points at >= 25 bps, zero at <= 5 bps")
    else:
        comp("costs", "Cost robustness (RA01)", None, "not assessed: no turnover or positions given")
    stab = frac_thirds if frac_years is None else 0.5 * frac_thirds + 0.5 * frac_years
    comp("stability", "Sub-period stability (RA05)", stab,
         f"{int(round(frac_thirds * 3))}/3 thirds positive" + (f", {frac_years:.0%} of years positive" if frac_years is not None else ""))
    cr = conc.get("concentration_ratio", float("nan"))
    comp("concentration", "PnL concentration (RA06)", _clip01((2.0 - cr) / 0.8) if cr == cr else 0.0,
         f"top-5% days carry {conc['top_5pct_share_of_pnl']:.0%} of PnL vs {conc.get('gaussian_expected_share', float('nan')):.0%} "
         f"for a normal distribution; worst of top-5%/top-1% ratios {cr:.2f}; full points at <= 1.2, zero at >= 2"
         if cr == cr else "total PnL is not positive")
    assessed = [c for c in comps if c["assessed"]]
    score = 100 * sum(c["points"] for c in assessed) / sum(c["weight"] for c in assessed)
    if implaus:
        score = min(score, 20)
    score = int(round(score))

    # ---------------- verdict
    artifact, missing, support = [], [], []
    if mu <= 0:
        artifact.append(f"The average {headline_label} return is not positive.")
    if dsr["dsr"] < DSR_FAIL:
        if n_eff > 1:
            artifact.append(f"Deflated Sharpe Ratio is {dsr['dsr']:.2f} (< {DSR_FAIL}): the best of {n_eff} zero-skill "
                            f"strategies would be expected to reach an annualized Sharpe of about {mt['expected_max_sharpe_ann']:.2f}, "
                            f"and this one ({perf['sharpe_ann']:.2f}) does not clearly beat that.")
        else:
            artifact.append(f"Probabilistic Sharpe Ratio is {dsr['dsr']:.2f}: the Sharpe is not distinguishable from zero.")
    if to is not None and not returns_are_net:
        c_check = float(costs_bps) if costs_bps is not None else DEFAULT_COST_BPS
        net_c = next(rw for rw in costs["sensitivity"] if rw["cost_bps"] == c_check)
        if net_c["mean_ann"] <= 0 < perf["gross_mean_ann"]:
            note = "" if costs_bps is not None else " (the default; pass costs_bps for your own cost)"
            artifact.append(f"Costs erase it: at {c_check:g} bps per unit of turnover{note} the net return is "
                            f"negative (break-even {costs['break_even_bps']:.1f} bps).")
    if implaus and perf["sharpe_ann"] > ARTIFACT_SHARPE:
        artifact.append(f"Implausible result ({', '.join(implaus)}): backtests this strong are often caused by "
                        "lookahead or a data error. Run check_code on the strategy.")

    if dsr["dsr"] >= DSR_PASS:
        support.append(f"Deflated Sharpe Ratio {dsr['dsr']:.2f} >= {DSR_PASS} with n_trials={n_eff}.")
    else:
        missing.append(f"Deflated Sharpe Ratio {dsr['dsr']:.2f} is below {DSR_PASS}.")
    if dsr["dsr"] < DSR_PASS and math.isfinite(trl_d):
        missing.append(f"Track record: about {trl_d:.0f} periods would be needed at this Sharpe (have {T}).")
    if not (perf["t_newey_west"] >= MIN_HAC_T):
        missing.append(f"The autocorrelation-robust (Newey-West, {L} lags) t-stat is {perf['t_newey_west']:.2f}, "
                       f"below {MIN_HAC_T:g}.")
    else:
        support.append(f"Newey-West t-stat {perf['t_newey_west']:.2f} ({L} lags).")
    if bench is not None:
        if not (bench["t_newey_west_alpha"] >= MIN_HAC_T):
            missing.append(f"No significant alpha over the benchmark (alpha {bench['alpha_ann']:.1%} a year, "
                           f"t {bench['t_newey_west_alpha']:.2f}, beta {bench['beta']:.2f}): the result can be "
                           "explained by benchmark exposure.")
        else:
            support.append(f"Significant alpha over the benchmark (t {bench['t_newey_west_alpha']:.2f}, "
                           f"beta {bench['beta']:.2f}).")
    else:
        missing.append(NO_BENCHMARK_REASON)  # CEO decision: without a benchmark the best verdict is "Some red flags"
    if frac_thirds == 1:
        support.append("Positive in all three sub-periods.")
    else:
        missing.append(f"Positive in only {int(round(frac_thirds * 3))} of 3 sub-periods.")
    if cr == cr and cr >= 2:
        warnings.append(f"PnL is concentrated (RA06): the best periods carry {cr:.1f}x the share a normal distribution "
                        "would give. Positively skewed strategies look like this too; check those days for data errors.")
    if n_trials is None:
        missing.append("n_trials was not declared.")
    if not costs["assessed"]:
        missing.append("Costs were not assessed (no turnover, positions, or returns_are_net).")
    if implaus:
        missing.append(f"Implausibly strong ({', '.join(implaus)}): check the code for lookahead.")
    missing += blockers + list(ctx.get("blockers", []))

    score_cap_note = None
    if artifact:
        extra = [m for m in missing if not m.startswith(("Implausibly", "Deflated Sharpe Ratio", "Track record"))]
        verdict, reasons = VERDICT_MAJOR_FLAGS, artifact + extra[:3]
    elif not missing:
        verdict, reasons = VERDICT_NO_FLAGS, support
    else:
        verdict, reasons = VERDICT_SOME_FLAGS, missing + support

    cap = {VERDICT_MAJOR_FLAGS: 40, VERDICT_SOME_FLAGS: 79}.get(verdict)
    if cap is not None and score > cap:
        score_cap_note = f"Score capped at {cap} because the verdict is '{verdict}' (raw weighted score {score})."
        score = cap
    assumptions += [
        f"Frequency: {freq_label} ({ppy:g} periods per year).",
        "Returns are simple returns; Sharpe ratios use excess-of-zero returns (no risk-free rate subtracted).",
        f"HAC bandwidth {L} lags: the larger of floor(4 (T/100)^(2/9)) and twice the run of significant positive "
        "autocorrelations, so overlapping returns are seen.",
        "Verdict thresholds: 'No red flags found' needs DSR >= 0.95, Newey-West t >= 2, all thirds positive, declared "
        "n_trials, assessed costs and, with a benchmark, significant alpha. 'Major red flags' is DSR < 0.50, a "
        "non-positive mean, costs erasing the edge at your cost, or an implausible Sharpe (> 5, daily or slower bars).",
        "The verdict is a statistical review of a hypothetical backtest, not a certification of a strategy.",
    ]
    sections = {"performance": perf, "costs": costs, "multiple_testing": mt, "stability": stability,
                "concentration": conc, "drawdown": drawdown}
    if bench is not None:
        sections["benchmark"] = bench
    sections.update(ctx.get("extra_sections", {}))
    if implaus:
        sections["implausibility"] = implaus
    if score_cap_note:
        sections["score_cap"] = score_cap_note
    return ReturnsAudit(verdict, reasons, score, comps, sections, warnings, assumptions,
                        tool=ctx.get("tool", "audit_returns"))

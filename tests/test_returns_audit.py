"""audit_returns on synthetic series: noise must not pass, a planted signal must, leaks must be caught."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from nba_auditor import DISCLAIMER, InputError, audit_returns
from nba_auditor.returns_audit import VERDICT_MAJOR_FLAGS, VERDICT_SOME_FLAGS, VERDICT_NO_FLAGS

IDX = pd.bdate_range("2012-01-02", periods=2520)


def _noise(seed: int, n: int = 2520, mu: float = 0.0, sd: float = 0.01) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n), index=IDX[:n])


def test_pure_noise_with_many_trials_has_major_flags():
    for seed in range(5):
        a = audit_returns(_noise(seed), n_trials=200, turnover=0.1)
        assert a.verdict == VERDICT_MAJOR_FLAGS, (seed, a.reasons)
        assert a.score < 50


def test_best_of_many_noise_strategies_is_never_no_red_flags():
    """Selection bias: pick the best of 100 noise strategies, declare n_trials=100.

    The best in-sample Sharpe of 100 zero-skill strategies over 5 years is about 1.1 annualized; the
    audit must not call it real. DSR of the selected strategy averages ~0.5, so a good share land in
    'Major red flags'. Without declaring n_trials the same strategies often look significant.
    """
    verdicts, naive_real = [], 0
    for seed in range(30):
        r = np.random.default_rng(1000 + seed).normal(0, 0.01, (1260, 100))
        best = r[:, np.argmax(r.mean(axis=0) / r.std(axis=0))]
        s = pd.Series(best, index=IDX[:1260])
        verdicts.append(audit_returns(s, n_trials=100, turnover=0.05).verdict)
        naive = audit_returns(s, n_trials=1, turnover=0.05)
        naive_real += naive.to_dict()["multiple_testing"]["deflated_sharpe_ratio"] >= 0.95
    assert VERDICT_NO_FLAGS not in verdicts
    assert verdicts.count(VERDICT_MAJOR_FLAGS) >= 8
    assert naive_real >= 20  # the uncorrected test is fooled most of the time


def test_unselected_noise_false_real_rate_is_low():
    reals = sum(audit_returns(_noise(500 + s, mu=0.0), n_trials=1, turnover=0.05).verdict == VERDICT_NO_FLAGS for s in range(100))
    assert reals <= 5


def test_planted_signal_scores_better_and_is_no_red_flags():
    rng = np.random.default_rng(11)
    planted = pd.Series(rng.normal(0.0009, 0.01, 2520), index=IDX)  # ~1.4 annualized Sharpe over 10 years
    market = _noise(99, mu=0.0003)  # independent benchmark: the planted edge is pure alpha
    a = audit_returns(planted, n_trials=10, turnover=0.05, benchmark=market)
    noise = audit_returns(_noise(11), n_trials=10, turnover=0.05, benchmark=market)
    assert a.verdict == VERDICT_NO_FLAGS, a.reasons
    assert a.score >= 85 and a.score > noise.score + 30
    assert a.to_dict()["multiple_testing"]["deflated_sharpe_ratio"] > 0.95


def test_planted_signal_some_flags_without_trials_or_costs():
    planted = pd.Series(np.random.default_rng(11).normal(0.0009, 0.01, 2520), index=IDX)
    a = audit_returns(planted)
    assert a.verdict == VERDICT_SOME_FLAGS
    joined = " ".join(a.reasons)
    assert "n_trials was not declared" in joined and "Costs were not assessed" in joined


def test_same_close_leak_from_phase0_has_major_flags():
    """Cross-sectional reversal ranked on today's return and credited with today's return (the leak),
    vs the honest version credited with tomorrow's return."""
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0003, 0.02, (1500, 200))
    k = 20
    picks = np.argsort(rets, axis=1)[:, :k]  # 20 worst performers today
    w = np.zeros_like(rets)
    np.put_along_axis(w, picks, 1.0 / k, axis=1)
    leaked = (w * rets).sum(axis=1)  # signal and fill on the same close
    honest = (w[:-1] * rets[1:]).sum(axis=1)  # held over the next day
    turnover = np.abs(np.diff(w, axis=0)).sum(axis=1).mean()
    a = audit_returns(pd.Series(-leaked, index=IDX[:1500]), n_trials=3, turnover=turnover, costs_bps=1)
    # (short the losers: the leak shows up as a huge positive Sharpe)
    assert a.verdict == VERDICT_MAJOR_FLAGS
    assert any("Implausible" in r for r in a.reasons)
    assert "implausibility" in a.to_dict()
    b = audit_returns(pd.Series(-honest, index=IDX[1:1500]), n_trials=3, turnover=turnover)
    assert b.verdict == VERDICT_MAJOR_FLAGS  # no edge, and 10 bps on ~1.8x daily turnover


def test_cost_grid_and_break_even_are_exact():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0004, 0.005, 1000), index=IDX[:1000])
    tau = 0.25
    a = audit_returns(r, turnover=tau, costs_bps=7)
    c = a.to_dict()["costs"]
    assert c["break_even_bps"] == pytest.approx(r.mean() / tau * 1e4, rel=1e-3)
    grid = {row["cost_bps"]: row for row in c["sensitivity"]}
    assert set(grid) == {0.0, 5.0, 7.0, 10.0, 20.0}
    assert grid[10.0]["mean_ann"] == pytest.approx((r.mean() - tau * 10 / 1e4) * 252, rel=1e-3)
    assert a.to_dict()["performance"]["mean_ann"] == pytest.approx((r.mean() - tau * 7 / 1e4) * 252, rel=1e-3)
    assert c["headline_basis"] == "net"


def test_costs_erase_edge_has_major_flags():
    r = pd.Series(np.random.default_rng(7).normal(0.0006, 0.01, 2520), index=IDX)
    a = audit_returns(r, turnover=2.0, n_trials=1)  # break-even ~3 bps, like our Phase 0 1-day model
    assert a.verdict == VERDICT_MAJOR_FLAGS
    assert any("Costs erase it" in x for x in a.reasons)


def test_positions_table_gives_turnover():
    dates = IDX[:40]
    pos = pd.DataFrame({"date": dates, "AAA": [0.5] * 20 + [0.0] * 20, "BBB": [0.5] * 40})
    r = pd.Series(np.random.default_rng(8).normal(0.001, 0.01, 40), index=dates)
    a = audit_returns(r, positions=pos.to_csv(index=False))
    c = a.to_dict()["costs"]
    assert c["turnover_per_period"] == pytest.approx((1.0 + 0.5) / 40)
    long = pos.melt(id_vars="date", var_name="ticker", value_name="weight")
    b = audit_returns(r, positions=long.to_json(orient="records", date_format="iso"))
    assert b.to_dict()["costs"]["turnover_per_period"] == pytest.approx(c["turnover_per_period"])


def test_benchmark_beta_and_excess():
    rng = np.random.default_rng(9)
    b = rng.normal(0.0004, 0.01, 2520)
    s = 0.0002 + 1.5 * b + rng.normal(0, 0.002, 2520)
    a = audit_returns(pd.Series(s, index=IDX), benchmark=pd.Series(b, index=IDX))
    bench = a.to_dict()["benchmark"]
    assert bench["beta"] == pytest.approx(1.5, abs=0.03)
    assert bench["alpha_ann"] == pytest.approx(0.0002 * 252, abs=0.03)
    assert bench["n_overlap"] == 2520


def test_input_formats_agree():
    r = _noise(10, n=300, mu=0.001)
    base = audit_returns(r).to_dict()["performance"]["sharpe_ann"]
    csv = r.rename("return").rename_axis("date").reset_index().to_csv(index=False)
    js_list = json.dumps(list(r.values))
    js_dict = json.dumps({str(k.date()): v for k, v in r.items()})
    js_records = json.dumps([{"date": str(k.date()), "ret": v} for k, v in r.items()])
    pct = r.rename("return").mul(100).rename_axis("date").reset_index().to_csv(index=False)
    equity = (1 + r).cumprod()
    equity = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), equity])
    for kwargs in ({"returns": csv}, {"returns": js_list}, {"returns": js_dict}, {"returns": js_records},
                   {"returns": pct, "units": "percent"},
                   {"returns": equity.rename("equity").rename_axis("date").reset_index().to_csv(index=False), "input_kind": "equity"}):
        got = audit_returns(**kwargs).to_dict()["performance"]["sharpe_ann"]
        assert got == pytest.approx(base, rel=1e-6), kwargs.keys()


def test_freq_auto_and_monthly():
    m = pd.Series(np.random.default_rng(12).normal(0.01, 0.04, 120), index=pd.date_range("2010-01-31", periods=120, freq="ME"))
    a = audit_returns(m, freq="auto")
    assert any("12 periods per year" in x for x in a.assumptions)
    b = audit_returns(m, freq="M")
    assert a.to_dict()["performance"]["sharpe_ann"] == b.to_dict()["performance"]["sharpe_ann"]


def test_report_has_required_sections_and_disclaimer():
    a = audit_returns(_noise(13, mu=0.0005), n_trials=5, turnover=0.1, benchmark=_noise(14))
    d = a.to_dict()
    for key in ("verdict", "reasons", "audit_score", "score_breakdown", "performance", "costs", "multiple_testing",
                "stability", "concentration", "drawdown", "benchmark", "disclaimer"):
        assert key in d
    assert len(d["stability"]["thirds"]) == 3 and len(d["stability"]["years"]) >= 9
    assert sum(c["weight"] for c in d["score_breakdown"]) == 100
    json.dumps(d)
    md = a.to_markdown()
    assert md.rstrip().endswith(DISCLAIMER)
    for heading in ("## Verdict", "## Costs", "## Multiple testing", "## Sub-period stability", "## Concentration",
                    "## Drawdown", "## Benchmark"):
        assert heading in md


def test_score_rescales_when_costs_not_assessed():
    a = audit_returns(_noise(15, mu=0.001))
    row = next(c for c in a.score_breakdown if c["check"] == "costs")
    assert row["assessed"] is False and row["points"] is None
    assert 0 <= a.score <= 100


def test_bad_inputs():
    with pytest.raises(InputError):
        audit_returns([0.01] * 10)
    with pytest.raises(InputError):
        audit_returns("date,a,b\n2020-01-01,1,2\n2020-01-02,1,2\n")
    with pytest.raises(InputError):
        audit_returns(_noise(1), freq="fortnightly")
    with pytest.raises(InputError):
        audit_returns(_noise(1), n_trials=0)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_percent_warning():
    r = _noise(16, mu=0.1, sd=1.0) * 30
    a = audit_returns(r)
    assert any("units='percent'" in w for w in a.warnings)


def test_concentration_flags_lottery_strategy():
    rng = np.random.default_rng(17)
    x = rng.normal(-0.0002, 0.003, 2520)
    jumps = rng.choice(2520, 25, replace=False)
    x[jumps] += 0.08
    a = audit_returns(pd.Series(x, index=IDX), n_trials=1, turnover=0.01)
    conc = a.to_dict()["concentration"]
    assert conc["concentration_ratio"] > 2
    # concentration is a warning and a score component, not a verdict blocker: positive skew is not an artifact
    assert any("concentrated (RA06)" in w for w in a.warnings)
    assert next(c for c in a.score_breakdown if c["check"] == "concentration")["points"] == 0
    assert math.isfinite(a.to_dict()["drawdown"]["max_drawdown"])


def test_score_caveat_everywhere():
    from nba_auditor import check_code
    from nba_auditor.catalog import SCORE_CAVEAT

    assert SCORE_CAVEAT == ("A score of 100 means none of our listed patterns matched, not that the code is free of "
                            "leaks. Static checks miss many leaks.")
    clean = check_code("import pandas as pd\nx = pd.read_csv('a.csv')\n")
    a = audit_returns(_noise(3, mu=0.0005), n_trials=5, turnover=0.1)
    for rep in (clean, a):
        assert rep.to_dict()["audit_score_note"] == SCORE_CAVEAT
        md = rep.to_markdown().splitlines()
        i = next(k for k, ln in enumerate(md) if ln.startswith("**Audit score:"))
        assert md[i + 2] == SCORE_CAVEAT


def test_verdict_labels_and_note_line():
    from nba_auditor.returns_audit import VERDICT_NOTE

    assert VERDICT_NOTE == ("No red flags doesn't mean a strategy will make money; it means these checks didn't "
                            "find the common errors.")
    for seed, mu in ((1, 0.0), (11, 0.0009)):
        a = audit_returns(_noise(seed, mu=mu), n_trials=10, turnover=0.05)
        assert a.verdict in ("No red flags found", "Some red flags", "Major red flags")
        assert a.to_dict()["verdict_note"] == VERDICT_NOTE
        lines = a.to_markdown().splitlines()
        i = lines.index(f"## Verdict: {a.verdict}")
        assert lines[i + 2] == VERDICT_NOTE  # right after the label

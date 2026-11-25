"""Markdown rendering for audit reports. JSON comes from each report's `to_dict()`."""

from __future__ import annotations

import math
from typing import Any

from .catalog import DISCLAIMER, SCORE_CAVEAT


def _pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x * 100:.{nd}f}%"


def _num(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def _score_table(rows: list[dict], cols: list[tuple[str, str]]) -> list[str]:
    out = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) if r.get(k) is not None else "n/a" for _, k in cols) + " |")
    return out


def code_audit_markdown(a) -> str:
    c = a.counts
    lines = ["# Backtest Auditor: code review", "",
             f"**Audit score: {a.score}/100** ({c['high']} high, {c['medium']} medium, {c['possible']} possible, {c['info']} info)",
             "", SCORE_CAVEAT,
             "", a.summary, ""]
    if a.score_breakdown:
        lines += ["## Score breakdown", "",
                  "Score = 100 minus penalties (high 25, medium 10, possible 3, info 0; at most 2 findings per rule count).", ""]
        lines += _score_table(a.score_breakdown, [("Rule", "rule_id"), ("Check", "title"), ("Findings", "findings"),
                                                  ("Worst", "worst_severity"), ("Penalty", "penalty")])
        lines.append("")
    if a.findings:
        lines += ["## Findings", ""]
        for f in a.findings:
            again = f", same issue as line {f.same_as}, not scored twice" if getattr(f, "same_as", 0) else ""
            lines += [f"### [{f.severity.upper()}] {f.rule_id} {f.title} (line {f.line}{again})", "",
                      f"```python\n{f.snippet}\n```" if f.snippet else "", "",
                      f"{f.message}", "", f"*Why it inflates results:* {f.why}", "", f"*Fix:* {f.fix}", ""]
    if a.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in a.notes] + [""]
    lines += ["---", DISCLAIMER]
    return "\n".join(lines)


def returns_audit_markdown(a) -> str:
    d = a.to_dict()
    p, c, mt, st, conc, dd = (d["performance"], d["costs"], d["multiple_testing"], d["stability"],
                              d["concentration"], d["drawdown"])
    title = "trade log" if a.tool == "audit_trade_log" else "return series"
    lines = [f"# Backtest Auditor: {title}", "", f"## Verdict: {a.verdict}", "", d["verdict_note"], "",
             "*A statistical review of a hypothetical backtest, not a certification of the strategy.*", ""]
    lines += [f"- {r}" for r in a.reasons] + [""]
    lines += [f"**Audit score: {a.score}/100**", "", SCORE_CAVEAT, "",
              "Weighted checks; points scale linearly between the stated thresholds. Checks that could not be "
              "assessed are left out and the rest rescaled to 100. The verdict caps the score: at most 40 for 'Major red flags' "
              "and 79 for 'Some red flags'." + (" Capped at 20 because the result looks implausible." if d.get("implausibility") else "")
              + (f" {d['score_cap']}" if d.get("score_cap") else ""), ""]
    lines += _score_table(a.score_breakdown, [("Check", "title"), ("Weight", "weight"), ("Points", "points"), ("Detail", "detail")])
    lines += ["", f"## Performance ({c['headline_basis']}{', ' + _num(c.get('headline_cost_bps'), 1) + ' bps' if c.get('headline_cost_bps') else ''})", ""]
    period = f" ({p['start']} to {p['end']})" if p.get("start") else ""
    lines += [
        f"| Metric | Value |", "|---|---|",
        f"| Periods | {p['n']} ({_num(p['years'], 1)} years){period} |",
        f"| Annualized return (arithmetic) | {_pct(p['mean_ann'])} |",
        f"| Annualized volatility | {_pct(p['vol_ann'])} |",
        f"| Sharpe ratio (annualized, naive) | {_num(p['sharpe_ann'])} |",
        f"| Sharpe ratio (Lo-style autocorrelation correction, {p['newey_west_lags']} lags) | {_num(p['sharpe_ann_lo'])} |",
        f"| t-stat: Newey-West ({p['newey_west_lags']} lags) / Lo-style / IID | {_num(p['t_newey_west'])} / {_num(p['t_lo'])} / {_num(p['t_iid'])} |",
        f"| Effective sample (T / long-run variance ratio) | {_num(p['effective_periods'], 0)} of {p['n']} (ratio {_num(p['variance_ratio'])}) |",
        f"| Skewness / kurtosis | {_num(p['skew'])} / {_num(p['kurtosis'])} |",
        f"| Hit rate | {_pct(p['hit_rate'])} |",
        f"| Lag-1 autocorrelation | {_num(p['lag1_autocorrelation'])} |", "",
    ]
    lines += ["## Costs (RA01)", ""]
    if c.get("sensitivity"):
        lines += [f"Turnover {_num(c['turnover_per_period'], 3)} per period ({_num(c['turnover_per_year'], 1)}x per year, one-way). "
                  f"**Break-even cost: {_num(c['break_even_bps'], 1)} bps** per unit traded.", "",
                  "| Cost (bps) | Ann. return | Sharpe | Total return |", "|---|---|---|---|"]
        lines += [f"| {_num(r['cost_bps'], 0)} | {_pct(r['mean_ann'])} | {_num(r['sharpe_ann'])} | {_pct(r['total_return'])} |"
                  for r in c["sensitivity"]]
    else:
        lines.append("Not assessed: pass `turnover` or `positions`." if not c["returns_are_net"] else
                     "Returns were declared net of costs; no turnover given, so no cost grid.")
    lines += ["", "## Multiple testing (RA03, RA04)", "",
              f"| Metric | Value |", "|---|---|",
              f"| Trials (n_trials) | {mt['n_trials']}{'' if mt['n_trials_declared'] else ' (not declared; assumed 1)'} |",
              f"| Expected best Sharpe of n_trials zero-skill strategies (annualized) | {_num(mt['expected_max_sharpe_ann'])} |",
              f"| **Deflated Sharpe Ratio** | **{_num(mt['deflated_sharpe_ratio'], 3)}** (pass >= 0.95) |",
              f"| Probabilistic Sharpe vs 0 | {_num(mt['probabilistic_sharpe_vs_0'], 3)} |",
              f"| Min track record vs the hurdle (periods / years) | {_num(mt['min_track_record_periods_vs_hurdle'], 0)} / {_num(mt['min_track_record_years_vs_hurdle'], 1)} |",
              f"| Min track record vs 0 (periods) | {_num(mt['min_track_record_periods_vs_0'], 0)} |",
              f"| Track record (periods) | {mt['track_record_periods']} |", "",
              f"Note: {mt['sr_variance_note']}.", ""]
    lines += ["## Sub-period stability (RA05)", "", "| Third | Period | Ann. return | Sharpe | NW t |", "|---|---|---|---|---|"]
    for i, t in enumerate(st["thirds"], 1):
        lines.append(f"| {i} | {t['start']} to {t['end']} | {_pct(t['mean_ann'])} | {_num(t['sharpe_ann'])} | {_num(t['t_newey_west'])} |")
    if st["years"]:
        lines += ["", "| Year | Return | Sharpe |", "|---|---|---|"]
        lines += [f"| {y['year']} | {_pct(y['return'])} | {_num(y['sharpe_ann'])} |" for y in st["years"]]
    lines += ["", "## Concentration (RA06)", ""]
    if conc.get("top_5pct_share_of_pnl") is not None:
        lines.append(f"The best {conc['top_periods']} periods (top 5%) carry {_pct(conc['top_5pct_share_of_pnl'], 0)} of total PnL; "
                     f"a normal distribution with the same mean, volatility and length would give about "
                     f"{_pct(conc['gaussian_expected_share'], 0)} (ratio {_num(conc['ratio_top_5pct'])}). The best 1% carry "
                     f"{_pct(conc['top_1pct_share_of_pnl'], 0)} vs {_pct(conc['gaussian_expected_share_1pct'], 0)} expected "
                     f"(ratio {_num(conc['ratio_top_1pct'])}). "
                     f"Without them, the summed return is {_pct(conc['sum_without_top_5pct'])}.")
    else:
        lines.append("Total PnL is not positive, so concentration is not meaningful.")
    if "ticker_concentration" in d:
        tc = d["ticker_concentration"]
        names = ", ".join(f"{x['ticker']} ({x['pnl']:,.0f})" for x in tc["top5"])
        lines += ["", f"By ticker (TL02): top 5 = {names}; share of PnL {_pct(tc['top5_share_of_pnl'], 0)}; "
                      f"PnL without them {tc['pnl_without_top5']:,.0f}."]
    lines += ["", "## Drawdown (RA07)", "",
              f"Max drawdown {_pct(dd['max_drawdown'])} (peak {dd['peak']}, trough {dd['trough']}). "
              f"Longest time under water: {dd['longest_underwater_periods']} periods ({_num(dd['longest_underwater_years'], 1)} years), "
              f"ending {dd['longest_underwater_end']}{' (still under water)' if dd['currently_underwater'] else ''}.", ""]
    if "benchmark" in d:
        b = d["benchmark"]
        lines += ["## Benchmark (RA08)", "", "| Metric | Value |", "|---|---|",
                  f"| Excess return (ann.) | {_pct(b['excess_mean_ann'])} |",
                  f"| Information ratio | {_num(b['information_ratio'])} |",
                  f"| NW t-stat of excess | {_num(b['t_newey_west_excess'])} |",
                  f"| Beta / alpha (ann.) / alpha NW t | {_num(b['beta'])} / {_pct(b['alpha_ann'])} / {_num(b['t_newey_west_alpha'])} |",
                  f"| Correlation | {_num(b['correlation'])} |",
                  f"| Benchmark Sharpe | {_num(b['benchmark_sharpe_ann'])} |", ""]
    if "trade_log" in d:
        tl = d["trade_log"]
        lines += ["## Trade log", "",
                  f"{tl['trades']} trades in {tl['tickers']} tickers, {tl['first_trade']} to {tl['last_trade']}; capital "
                  f"{tl['capital']:,.0f}; fees in log {tl['fees_total']:,.0f}; marked with daily prices: "
                  f"{'yes' if tl['marked_with_prices'] else 'no'}.", ""]
        sh = d["survivorship_hints"]
        lines += ["### Survivorship hints (TL01, heuristic)", ""]
        lines += [f"- {h['message']}" for h in sh["hints"]] or ["- None found."]
        lines += [f"- {sh['note']}", ""]
    if d["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in d["warnings"]] + [""]
    if d["assumptions"]:
        lines += ["## Assumptions", ""] + [f"- {x}" for x in d["assumptions"]] + [""]
    lines += ["---", DISCLAIMER]
    return "\n".join(lines)

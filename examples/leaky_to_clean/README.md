# From leaky to clean: a walkthrough

A short momentum strategy, written the way a chat assistant often writes one, run through the Backtest Auditor,
fixed, and audited again. Everything runs locally on **synthetic prices** generated inside the script, with no
vendor data, no network and no API key. All results here are hypothetical backtests on made-up data.

| File | What it is |
|---|---|
| [`leaky_strategy.py`](leaky_strategy.py) | The strategy with four common bugs, plus a parameter sweep that picks the "best" setting |
| [`fixed_strategy.py`](fixed_strategy.py) | The same idea with the bugs fixed |
| [`walkthrough.py`](walkthrough.py) | Runs `check_code` and `audit_returns` on both and writes the reports |
| [`output/`](output/) | The four reports, exactly as the auditor printed them |

Run it from this folder, in an environment where the auditor is installed (see [the quickstart](../../docs/quickstart.md)):

<!-- test: run -->
```bash
python walkthrough.py
```

CI runs the same script (`tests/test_docs_snippets.py`) and checks that the results below still hold.

## The setup

The prices are a **random walk**: by construction, nothing in the past predicts the future. Any edge a backtest
finds on this data comes from a bug. That makes it a fair test of the auditor, because we know the right answer.

## Step 1: the leaky version

`leaky_strategy.py` computes 5-to-60-day momentum, turns it into a z-score, goes long above a threshold and short
below it, and sweeps 15 combinations of lookback and threshold. The best one shows a hypothetical annualized Sharpe
ratio of **6.77** (gross of costs, synthetic data). On a random walk, that can only come from bugs.

The bugs:

1. **Same-bar signal and return.** `df["signal"] * df["ret"]` earns today's return with a position decided from
   today's close, which already contains that return.
2. **Full-sample z-score.** `mom.mean()` and `mom.std()` use the whole history, including the future.
3. **The sweep is scored on all the data**, and only the winner is reported.
4. **No trading costs.**

### What the auditor said

**`check_code`** ([full report](output/1_leaky_check_code.md)):

Audit score 52/100.

| Rule | Severity | Line | Finding |
|---|---|---|---|
| LA09 | medium | 25 | Global normalization over all rows (the full-sample z-score) |
| LA07 | high | 27 | Signal and fill on the same bar |
| LA11 | medium | 34 | Parameters optimized on the full sample ("It tries about 15 combinations: pass n_trials>=15 to audit_returns.") |
| CO01 | possible | 27 | No transaction costs (a Sharpe ratio is computed from the cost-free returns; since v0.3 costs are a caveat, never above possible) |

It caught all four bugs here (the missing costs only as a "possible" caveat). Version 0.1 missed bug 2, because the z-score sits inside a function that takes `df`
as a parameter (fixed in 0.2). Don't read too much into a clean sweep on one example: the
[Known limitations](../../docs/LIMITATIONS.md) list what the checks miss, the published test results at
https://nothingbutalpha.app/backtest-audit give the measured miss rate on code the lint hasn't seen, and a clean
`check_code` result is still weak evidence.

**`audit_returns`** with `n_trials=15`, average turnover and a 10 bps default cost, against the asset itself as
the benchmark ([full report](output/2_leaky_audit_returns.md)):

> **Verdict: Major red flags.** Implausible result (annualized Sharpe of 6.2): backtests this strong are often
> caused by lookahead or a data error. Run check_code on the strategy.

The return audit doesn't need to see the code to know something is wrong. Every other statistic in that report
(a Deflated Sharpe Ratio of 1.00, all three sub-periods positive) looks excellent, which is exactly why those
statistics alone can't be trusted when the data has a leak.

## Step 2: the fixed version

`fixed_strategy.py` changes four things:

1. `position = signal.shift(1)`: decide at today's close, earn tomorrow's return.
2. The z-score uses a trailing 252-day window.
3. One setting (20-day lookback, threshold 1.0), and `n_trials=15`, because those 15 variants were already tried
   in step 1. Dropping the sweep from the code doesn't erase it from your history.
4. It reports turnover, and the auditor charges 10 bps per unit of turnover.

### What the auditor said

**`check_code`** ([full report](output/3_fixed_check_code.md)): **no findings**, score 100. Read that with the
caveat printed next to it: *a score of 100 means none of our listed patterns matched, not that the code is free
of leaks.*

**`audit_returns`** ([full report](output/4_fixed_audit_returns.md)):

> **Verdict: Major red flags.**
> - Deflated Sharpe Ratio is 0.08 (< 0.5): the best of 15 zero-skill strategies would be expected to reach an
>   annualized Sharpe of about 0.89, and this one (0.17) does not clearly beat that.
> - The autocorrelation-robust (Newey-West, 6 lags) t-stat is 0.34, below 2.
> - No significant alpha over the benchmark (alpha 2.0% a year, t 0.37, beta 0.03).
> - Positive in only 2 of 3 sub-periods.

## What this shows, and what it doesn't

- **The fix didn't make the strategy profitable. It removed a phantom edge.** On a random walk, the honest result
  is "nothing here", and that's what the audit reports. Fixing leaks usually makes a backtest look worse.
- **A clean code audit is not a clean bill of health.** The lint caught these four bugs, but it misses others
  in code it hasn't seen. Read the code yourself; the auditor is a second pair of eyes, not the only one.
- **Even "No red flags found" wouldn't mean the strategy makes money.** It means these checks didn't find the
  common errors. It says nothing about the future, about costs you didn't declare, or about variants you tried
  but didn't count in `n_trials`.

---
*These are automated statistical and code reviews of hypothetical backtests on synthetic data. They are general
information, not investment advice, not tailored to anyone's financial situation or objectives, and do not predict future results. Nothing But Alpha
is not a registered investment adviser.*

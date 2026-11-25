# Backtest Auditor reference

Tools, prompts, rules, verdicts and the command line, for version 0.7.0. This text is taken from the package README;
only link paths, one budget phrase, one heading and one reference to our internal issue tracker were changed, and a
link to the walkthrough was added. The limits in the
[README](../README.md) apply to everything here: the checks catch some common errors, miss many, and can flag
correct files, even plain ones.

## Tools

| Tool | Input | What you get |
|---|---|---|
| `check_path(path, include_glob?, max_files?)` | A `.py` file, notebook or folder on disk | Per-file findings with rule id, severity, title, exact line range and a one-line "why it matters"; suggested fixes where a minimal edit is unambiguous (see below). Read-only |
| `audit_backtest(code_path?, returns_csv?, trades_csv?, benchmark_csv?, n_trials?, turnover?, ...)` | Paths to the code and the results | Every audit that applies, and **one combined verdict** (the worst of the parts) with reasons per part |
| `check_code(source)` | Python/pandas strategy code | Findings with rule id, severity, line, why the problem inflates results, and a fix. Score 0-100. Also `suggested_fixes` |
| `audit_returns(returns, freq, costs_bps?, turnover?, positions?, benchmark?, n_trials?, ...)` | A return series (CSV/JSON) | Cost grid and break-even cost; Sharpe with Newey-West and Lo t-stats; **Deflated Sharpe Ratio**; minimum track record length; thirds and years; top-day concentration; drawdown; IR/beta. Verdict **No red flags found / Some red flags / Major red flags** with reasons, plus a score |
| `audit_trade_log(trades, prices?, initial_capital?, ...)` | date, ticker, side, qty, price[, fee] | Daily PnL and turnover from the log, then `audit_returns`, plus ticker concentration and survivorship hints |
| `explain(check_id)` | e.g. `LA07`, `RA03`, or empty | What the check does, why it matters, how to fix it, references |

Every tool returns `{"markdown": ..., "report": {...JSON...}}`. The server also offers three MCP prompts, which
clients show as one-click commands: **Find look-ahead bias in this repo** (`find_lookahead_bias`), **Audit this
backtest before I rely on it** (`audit_backtest_before_trusting`) and **Draft a backtest from my idea**
(`draft_backtest`, see below).

### For agents: `check_path`, suggested fixes and `audit_backtest`

Coding agents (Claude Code, Cursor, Claude Desktop) call the server while they write backtests and act on what it
returns, so these tools favor precision: few findings get an edit, and an edit is only offered when the change is
unambiguous.

- **`check_path` is read-only.** Files are parsed with `ast` and never imported or run. Only `.py`, `.pyw` and
  `.ipynb` files are read, whatever `include_glob` says. Virtual environments (any folder with `pyvenv.cfg`),
  `site-packages`, `node_modules`, hidden folders (`.git`, `.venv`, ...), build and cache folders are skipped. Folder
  symlinks and junctions are not followed, so link loops end; a file symlink is read only if it points inside the
  folder. Limits: 200 files by default (`max_files`, at most 2,000), 1 MB per file, 20 MB in total. Binary,
  unreadable and undecodable files are listed under `skipped` with the reason. A UTF-8 byte-order mark is stripped
  before parsing; other encodings are used only when the file declares them (PEP 263). Notebook findings carry
  the cell and the line in the cell.
- **Suggested fixes** are offered only for high or medium findings in these unambiguous cases:

  | Rule | Pattern | Suggested edit |
  |---|---|---|
  | LA01 | `y = x.shift(-k)` alone, with `x` a plain column or variable, never combined with `x` later | `x.shift(k)`; or just `x` when the result is already lagged later (e.g. `position.shift(1)`), so the fix never lags twice |
  | LA02 | `rolling(..., center=True)`, not feeding a label | `center=False` (trailing window) |
  | LA03 | `.bfill()`, `.backfill()`, `fillna(method="bfill")` on rows visibly sorted ascending (a resample, `sort_index()`, or `sort_values` ending on a date); `reindex(..., method="bfill")` on the row index | The forward-fill version |
  | LA05 | A scaler or model fitted on all rows, with a visible `X[:k]`/`X.iloc[:k]` split later | Fit on the training slice |
  | LA06 | `train_test_split` without `shuffle=False`; `KFold(shuffle=True)`; only when the rows split are visibly ordered by date alone and the file shows no sign of panel data (an asset column such as `ticker`, any `groupby`, a sort with another key before the date) | `shuffle=False`; `TimeSeriesSplit` (plus its import) |

  Each fix is a structured edit (`file`, `start_line`, `end_line`, `original`, `replacement`) with a one-line
  rationale. Before a fix is returned, the server applies it in memory and lints again: the finding must be gone,
  the code must parse, and no new high or medium finding may appear. Otherwise the fix is dropped. No fix is offered
  where the mechanical edit would change what the code means: `x.shift(-1) > x` or `x.shift(-1) / x - 1` (a lead
  combined with `x`), `pct_change(-k)` and `diff(-k)` (the mirror has the opposite sign), a lead lagged again later,
  a column later used as a model's label (y of `fit`, a split, a metric, or a label-named column), fills across
  columns (`axis=1`), fills on data whose ascending order isn't visible (bfill on newest-first data reads the past,
  so ffill would be the leak). Also none for `stratify=`, `interpolate()`, more than one candidate, or notebooks.
  Fix checks are budgeted (12 per file, 60 per call, 10 seconds per call); fixes past the budget are skipped and counted
  in `fixes_not_checked`. Most
  findings get no fix; an agent should explain those rather than guess an edit. A fix removes a pattern; it doesn't
  make a backtest trustworthy.
- **`audit_backtest`** runs `check_path` on the code, `audit_returns` on the returns and `audit_trade_log` on the
  trades, each against the benchmark if given, and returns the worst verdict of the parts, in the same language,
  always followed by the caveat line. A high code finding is "Major red flags"; a medium one is "Some red flags". The
  caps stay: without a benchmark, or with code only, the best verdict is "Some red flags".
- **Eval (in-sample; not an accuracy measure).** [`eval/agent_fix`](../eval/agent_fix/README.md) applies the
  suggested fixes to small leaky and clean repos and counts leaks removed, new findings and clean lines changed. We
  wrote both the repos and the fix rules, so the result only shows that the fixes behave as designed on patterns we
  already knew; it says nothing about how often they help, or what they change, in your code. On those 27 files the
  fixes removed 11 of 23 planted leaks, added no new findings, changed no clean lines and changed none of the lines
  where a mechanical fix would change the meaning (the `meaning_preservation` repo, from the red-team review). The
  12 not removed get no suggested fix by design.

### Draft a backtest with your AI assistant (`draft_backtest`, new in 0.7.0)

You describe a strategy idea of your own in plain English, for example *"backtest buying SPY when RSI(14) drops
below 30 and selling above 50, in vectorbt"*. The `draft_backtest` prompt (arguments `idea`, required, and `format`:
`pandas` by default, `vectorbt` or `backtrader`) asks your AI assistant to:

1. restate the rules and ask you about anything ambiguous (universe, dates, entry and exit, sizing, costs);
2. write the code from our template for that format, keeping its timing: signals use data up to the bar's close,
   trades fill at the next bar's open, costs are a parameter (never zero by default), and train and test are split
   by date;
3. run `check_code` on the finished code and fix every high and medium finding before showing you any result;
4. present any result as hypothetical, simulated past performance that is not a prediction, and say that the check
   misses many leaks.

The server doesn't write or run the code, and makes no network calls: your assistant writes it, and you run it on
your own data. The ideas are yours; nothing here is investment advice or a view on any strategy. A step-by-step
example, from idea to audit and fix: [docs/walkthrough-draft-to-audit.md](walkthrough-draft-to-audit.md).

The templates are small and commented (one asset, daily bars, long or flat, data from a CSV you supply), and each
saves its net returns to `returns.csv` for `audit_returns`. They are MCP resources, and the prompt also carries the
chosen template in full for clients that don't read resources:

| Format | Resource | Command line |
|---|---|---|
| pandas | `nba-auditor://templates/pandas` | `backtest-auditor template pandas` |
| vectorbt | `nba-auditor://templates/vectorbt` | `backtest-auditor template vectorbt` |
| backtrader | `nba-auditor://templates/backtrader` | `backtest-auditor template backtrader` |

Tests (`tests/test_draft_backtest.py`) check that each template lints with zero findings and that a broken variant
of each is flagged (a pandas signal that isn't shifted, unshifted vectorbt signals filled at the close, a
backtrader strategy reading the next bar or trading on the close). Two gaps found while building the templates
are flagged since lint v0.7: vectorbt signals that aren't shifted but fill at `price=open` (the same bar's open), and
a signal returned by a separate helper function and then traded at the open. The prompt still asks the assistant to
confirm by hand that every signal is shifted one bar before it trades, because the check misses many leaks.
A clean check on a template-based strategy is weak evidence, not proof that it is free of leaks.

### Code rules (`check_code`)

| id | Rule | Typical severity |
|---|---|---|
| LA01 | Negative `shift`/`pct_change`/`diff` in a feature, signal or condition (labels are fine). v0.6: a lag helper called with a negative lag (`lag(r, -2)` with `def lag(s, k): return s.shift(k)`); a helper that returns a negative shift is also shown at its `return` line | high |
| LA02 | `rolling(..., center=True)`; rolling/`pct_change`/`shift` on data sorted newest-first (`sort_values('date', ascending=False)`, `sort_index(ascending=False)`). v0.6: two-sided transforms (`filtfilt`, `sosfiltfilt`, `hpfilter`, `savgol_filter`, `gaussian_filter1d`, `uniform_filter1d`, `np.gradient`, `argrelextrema`/`argrelmin`/`argrelmax`, `find_peaks`, `convolve(..., mode="same")`, a Kalman `.smooth(...)`, `seasonal_decompose`, `STL`, `lowess`, `bkfilter`, `cffilter`) and `rolling(center=<setting>)` when every visible definition of the setting is `True`: high when the result reaches a signal, position, PnL or model input (followed through assignments and helper return values), possible otherwise; not reported for a label, or when re-run on data up to each date inside a loop | high / possible |
| LA03 | `bfill`, `fillna(method="bfill")`, `reindex(method="bfill")`, `interpolate()` | medium / possible |
| LA04 | `merge_asof(direction="forward"/"nearest")` | high / medium |
| LA05 | v0.7 (medium): a scaler, imputer, selector or `polyfit` fitted inside a helper on its whole argument when a call site passes the full sample once (not per window, per date or a training slice) and the result reaches a model that is later split or cross-validated, a position or PnL; features chosen on all rows (`corrwith(y)`, `SelectKBest`, `mutual_info_*`, `f_regression`) and then cross-validated or walk-forwarded; pairs kept by a full-sample `coint`/`adfuller` test and backtested on the same rows; weights from the full-sample covariance applied to the same returns; a grid search fitted on all rows and then scored on a slice of them. Earlier: scaler/PCA/model fitted on all rows before the time split, or fitted and evaluated in-sample; a scaler fitted on all rows while the model trains on a subset of them (medium). A fit on rows bounded above (`.loc[:cut]`, `.iloc[:k]`, `[:k]`, `.loc[a:b]`, `df.date < cut`, `between`) is a training subset and is not flagged; a start filter (`.loc['2000':]`, `df.index >= '2000'`) or a test-window mask is not a training subset. v0.6: a full-sample estimate (`sm.OLS`/`WLS`/`RLM`, `np.polyfit`, `linregress`, `coint`, `KMeans`, `GaussianMixture`, `GaussianHMM`, `arch_model`, `ARIMA`, `PCA`, ...) outside loops and helpers, in a file with no split, training-window or walk-forward names, whose output reaches a position or signal that is booked against returns (medium; not for forecasts compared with outcomes, `fit(last_obs=...)`, or positions computed for live use with no backtest in the file) | high / medium |
| LA06 | v0.7 (medium): `cross_val_score`/`cross_validate`/`cross_val_predict` with integer, default or unshuffled k-fold folds when the label is a forward return (computed with a negative shift or named like `ret_fwd_20`); a hand-written walk-forward loop whose training slice ends at the first test row while the label looks 2 or more bars ahead; `.sample(frac=1)` returned by a helper and then split. Post-hoc: a shuffled k-fold with a forward-named label in a file with no time-series hint; the k-fold rule also on a 1-bar forward label (P6). Earlier: `train_test_split` without `shuffle=False`; shuffled or default k-fold CV on time series. v0.6: `TimeSeriesSplit` without `gap` when a label in the file looks 2 or more bars ahead (medium, unless the folds are trimmed or the file mentions purging or an embargo); `fit_resample` (SMOTE and similar) on all rows before a split or cross-validation (high); `.sample(frac=1)` before a time-series split, and torch `random_split` on time series (medium); `KFold(shuffle=True)` also when the only time-series hint is a forward-return name (medium) | high / medium / possible |
| LA07 | Signal from bar t's close times bar t's return, including numpy/list slices (`c[1:]` vs `c[:-1]`, `zip`, `np.diff`), inline conditions and helper functions; a close-based signal times a return that starts at bar t's open; bar loops that decide on bar i's close (also through a variable set from it earlier in the iteration, v0.4) and fill at bar i's open (high) or close (medium) or book bar i's return; an entry or exit priced at bar t's open and selected by a signal from bar t's close, high, low or volume: `open.where(sig)`, `open.mask(~sig)`, `open[sig]`, `df.loc[sig, "open"]`, `np.where(sig, open, nan)`, `df.loc[sig, "entry_px"] = df["open"]` (v0.4; high when the result has a fill name such as `entry_px`, `buy_price`, `exit_px`, else possible). v0.5: a positive shift at the end of a function passed to `groupby(...).transform`/`apply`/`pipe` (`lambda s: s.pct_change(7).shift(1)`, or a helper returning `param....shift(k)`) lags the result, and a frame used only to group (`panel.groupby("date")["mom"]`) doesn't count against a lag; `cheat_on_close`/`set_coc`/`trade_on_close`; vectorbt `from_signals(price=close)` (medium) or default close fills (possible) with unshifted entries. v0.6: a fill or entry priced at the same bar's low or high (`entry_px = bar["low"]` in a bar loop, high; `fill_px = np.where(sig, low, nan)`, `from_signals(price=low)`, medium); a limit set from bar t's close checked against bar t's own low or high (high); a bar's VWAP is treated like its open for same-bar timing (a close signal entering at that bar's VWAP) | high / medium / possible |
| LA08 | Label is the same-bar return (not shifted forward) | possible |
| LA09 | v0.7 (medium): a whole-sample mean/std/quantile/median (or a per-asset `groupby(...).apply`/`transform`) of a helper's or constructor's argument, combined with that data, when a call site passes the full sample and the result reaches a model, signal or PnL (metric helpers such as R² or a volatility are not transforms); `groupby(level=<asset>).rank(pct=True)`; `percentileofscore` against the full history; pooled `qcut`/`quantile` winsorizing across all dates of a panel that is evaluated date by date; full-history `rank(pct=True)` of a dated series reaching a position; an all-time high/low driving a signal. Earlier: global z-score / min-max / `x / x.abs().max()` / `x / x.std()` over all rows, also inside functions and helpers; a full-sample mean/median/quantile used as a signal threshold; full-column `rank`, `zscore`, `qcut` feeding a signal; per-asset `groupby().transform` of a full-history statistic; a lambda, a named helper or `zscore` passed to `apply`/`transform`/`pipe` that scales each column by its own whole-column statistics (v0.4; not flagged with `axis=1`, `groupby`, rolling/expanding windows, a transposed frame, a training slice or one date's cross-section). Medium when the result feeds a signal and the column is a time series; possible otherwise. v0.6: `groupby(<asset>)[col].rank(pct=True)` over each asset's whole history (medium when it reaches a signal); winsorizing at quantiles of every row (`quantile(0.02)` then `clip`) or `fillna(frame.mean()/median())` before the frame is split into train and test rows (medium) | medium / possible |
| LA10 | Feature or signal built from the label; `X` that drops `target` but keeps `fwd_ret` (drop lists, constants, `df.columns` comprehensions and `str.startswith` masks are resolved per column); forward columns passed to `fit` or a training dataset, also inside a frame built with `pd.DataFrame({...})`/`pd.concat`; a neutral-named forward value that reaches a signal or position; an outcome (a forward average, a forward PnL statistic, a target encoding) assigned to a signal, position or weight name, or transformed (rolling, sign, rank, clip, ratio) and multiplied with a return (`*`, `.mul`, `np.dot`, `np.where(c, r, -r)`, `np.copysign(r, s)` with `s` from future data, v0.4); a forward column reshaped with `pivot`/`pivot_table`/`unstack` keeps its forward flag and lead (v0.4); literal `query`/`eval` strings are read as the code they stand for (v0.4); a strategy-named series kept only on rows chosen by an inline comparison on future data (`strat = bench[mkt_fwd > 0]`, medium, v0.4); buckets formed from the forward return and then used to average it (medium). Not flagged: forward returns used only as the outcome (averaged per date or bucket, an IC, booked against a position decided earlier, next-open fills, a bucket `pivot_table` of the forward return, `fw[mask].mean()` with a mask from past data); a lead undone by a lag of the same size (`shift(-H)` then `shift(H)`); a fill model, where a resting limit order placed without future data fills if the next bar's low (high) reaches the limit and that mask only selects trade returns (v0.4). A neutral name that the lint can't tie to a model input or signal is "possible" | high / medium / possible |
| LA11 | Parameter grid (loops, `itertools.product`, dict/list comprehensions) scored on the full sample, best kept (reports the grid size to use as `n_trials`). v0.6: also shown at the line that keeps the best (`best = max(scores, key=scores.get)`), as a repeat that is scored once | medium |
| LA12 | Intraday `resample` stamped at the bar start (default or `label='left'`): medium when the bars are joined, reindexed or assigned back onto another time index, possible otherwise. A resampled column assigned into a frame of the same bars (`bars["volume"] = ticks["size"].resample("1min").sum()`) is not "another time index" (v0.4). v0.6: names computed only from bars of the same rule (`feat = pd.DataFrame({... a_1s.rolling(5).mean()})`, `nxt = mid_1s.pct_change().shift(-1)`) keep those bars' stamps, so joining them is not a realignment; `groupby(<period of the index>)[col].transform("last"/"max"/"sum"/"mean"/"std"/...)` or `pd.Grouper(freq=...)` writes a period's final value onto the earlier bars of that period (medium when it reaches a signal, possible otherwise; `first`, `count`, running statistics and a following `.shift` are fine) | medium / possible |
| LA13 | `x[i+1]`, `x.iloc[i+1]` or a slice starting after `i` inside an `if` in a `for` or `while` bar loop, or a value read from a later bar earlier in the loop. v0.4: offsets from module constants and parameter defaults are resolved (`c[i + K]` with `K = -2` is the past); string character loops and loops whose outputs only ever become a label or target are not reported; a later bar read only for a missing-data check (`np.isnan`, `pd.isna`, `is None`) is possible. v0.5: a resting order simulated in a loop is not reported: an order decided at bar `i` from past data (an enclosing condition, an `and` term, or an `elif` of another fill check), filled if bar `i+1`'s low or high reaches its level (a level known before that bar's high and low: past data or bar `i+1`'s open), with the fill or exit priced from that level and every later-bar read in the branch a price assigned to a fill, exit or PnL name. Possible when the fill isn't priced from the level. v0.6: in a backtrader `Strategy`/`Indicator`, a positive line index (`self.data.close[1]`) is the next bar (high); a trade list read only for its size (`len(trades) >= 50`, `if trades`, `not trades`) is not a decision on fill values | high / possible |
| LA14 | Post-hoc v0.7: also shown at the test-scoring call inside a selection loop (a repeat scored once); a dict comprehension of test-set scores picked with `max(d, key=d.get)` (high). Earlier: early stopping or best-model selection on the test set (`eval_set`/`valid_sets`/`validation_data` with early stopping); a loop that picks the model, parameter or feature with the best test score. v0.6: `max`/`min` with a `key` that scores test data (`max(grid, key=lambda t: f1_score(y_test, p > t))`); a dict filled in a loop with one test-set score per candidate (test-named data or rows `~train_mask`) and picked with `max(d, key=d.get)` (high; a repeat scored once when the loop is already reported); an Optuna-style objective passed to `optimize` that returns a score on test data (high) | high / medium |
| LA15 | Fundamentals merged onto prices by fiscal period end (`period_end`, `fiscalDateEnding`, `datadate`, `calendardate`, ...) instead of the filing date, with no visible reporting lag: `merge_asof` high, period-end index joined medium, exact merge medium if forward-filled, else possible | high / medium / possible |
| SB01 | Today's constituents: `read_html(<Wikipedia S&P 500>)`, `tickers_sp500()`, snapshot CSVs, hard-coded ticker lists; filtering a panel to names whose last date is the panel's last date (medium). v0.6 (medium): `dropna(axis=1)` (or `thresh=len(...)`) on a price- or return-named wide frame that is then run through returns or rolling windows; columns kept by `iloc[-1].notna()`; a universe taken from the last row (`iloc[-1].nlargest(n)`, not of a score or signal) or from whole-period returns (`px.iloc[-1] / px.iloc[0]`) that then selects the backtest's columns; members whose `end`/`removed` date is empty (not combined with a point-in-time test); `groupby(id).filter(lambda g: g[date].max() >= <fixed end date>)`. v0.7 (medium): `dropna(axis=1)` reached through `.loc[...]`, `.pipe(lambda d: ...)` or a resample (not on a trailing or rebalance-dated window, and not when the kept frame is then cut into formation/trading windows); `cols[F.iloc[-1].notna()]` and a universe cut from the last row by a threshold (`last[last > x].index`) or by `last_valid_index`; empty end/removed/delist dates inside a helper or a `query("...isnull()")` string; a constituents file named as a current snapshot (`current`, `latest`, `today`) whose names select a backtested frame. A list reduced to the last row (today's orders) is not a backtest | high / medium / possible |
| DQ01 | yfinance data caveats | info |
| DQ02 | Raw `Close` returns with `auto_adjust=False`; `Adj Close` mixed with raw OHLC | medium / possible |
| CO01 | No commissions, fees, spread or slippage anywhere (vectorized or loop PnL); engines called without fees. A realism caveat, not a lookahead error, so never above "possible" (since v0.3); the message says when a Sharpe, CAGR or total return is computed from the cost-free returns | possible |

The lint uses `ast` plus a light data-flow pass over assignments: it tracks whether each column is
forward-looking, lagged, price-derived, a return or a signal, and uses name heuristics (`target`/`fwd_*`/`y`
mean labels, `signal`/`position`/`pred` mean signals). When unsure it says "possible". Code that doesn't
parse falls back to a few text patterns, all reported as "possible".

### Return checks (`audit_returns`)

- **RA01** costs and the break-even cost.
- **RA02** Sharpe significance: Newey-West, plus a Lo-style (2002) correction.
  - The HAC bandwidth widens automatically when returns overlap.
  - All significance tests use an effective sample size: T divided by the long-run variance ratio.
- **RA03** Deflated Sharpe Ratio (Bailey & López de Prado 2014).
- **RA04** MinTRL (Bailey & López de Prado 2012).
- **RA05** sub-period stability.
- **RA06** concentration in the best 5% and 1% of days, compared with a normal distribution.
- **RA07** drawdown.
- **RA08** benchmark: excess return, IR, beta, and alpha with its t-stat.
- **RA09** implausibility: a Sharpe above 3, or a daily hit rate above 65%, on daily or slower bars.

Trade logs add **TL01** survivorship hints and **TL02** concentration by ticker.

**Verdict rules** (always a statistical review of a hypothetical backtest, never a certification). Every verdict is followed by this line: *No red flags doesn't mean a strategy will make money; it means these checks didn't find the common errors.*
- *Major red flags*: any of the following.
  - DSR < 0.50, meaning the result does not beat what the best of `n_trials` zero-skill strategies would show.
  - The mean return is not positive.
  - Costs erase the edge at *your* declared cost (10 bps if you gave turnover but no cost).
  - The Sharpe is above 5 on daily or slower bars.
- *No red flags found*: all of the following.
  - DSR >= 0.95 and a Newey-West t of at least 2, both with the autocorrelation adjustment.
  - All three thirds are positive.
  - `n_trials` is declared and costs are assessed; a user-supplied `trials_sr_variance` is not below the
    zero-skill sampling variance.
  - The series has dates (or an explicit `freq`), regularly spaced, with finite values.
  - A benchmark is supplied and the alpha over it is significant (t >= 2). Without a benchmark the best verdict
    is "Some red flags", with the reason "No benchmark supplied: market exposure can't be separated from skill."
  - The result is not implausible.
  - The series is not intraday, and the units look right.
  - For trade logs: daily prices were given, fills match those prices, and no anachronistic ticker was found.
- *Some red flags*: everything else. Concentration (RA06) is a warning and a score component, not a blocker,
  because positively skewed strategies (trend, long volatility) are concentrated by nature.

**Score**: *A score of 100 means none of our listed patterns matched, not that the code is free of leaks. Static checks miss many leaks.* (Every report prints this next to the score.) Weighted points for DSR (40), costs (20), stability (20) and concentration (20). A check that can't
be assessed is dropped and the rest rescaled. The verdict also caps the score: at most 40 for *Major
red flags*, 79 for *Some red flags*, and 20 if the result is implausible. The JSON has the full breakdown. MinTRL is
reported but not scored separately, because "track record >= MinTRL vs the hurdle" is the same test as DSR >= 0.95.


## Use as a library or command line

```python
from nba_auditor import check_code, check_path, audit_backtest, audit_returns, audit_trade_log, explain

rep = check_code(open("strategy.py").read())
print(rep.to_markdown())               # or rep.to_dict() for JSON

res = check_path("strategies/")        # dict: per-file findings, line ranges, suggested_fixes, skipped
res = audit_backtest("strategy.py", "returns.csv", benchmark_csv="spy.csv", n_trials=40, turnover=0.4)
print(res["verdict"], res["verdict_note"])

rep = audit_returns("returns.csv", freq="D", turnover=0.4, n_trials=40, benchmark="spy.csv")
print(rep.verdict, rep.reasons, rep.score)
```

```bash
python -m nba_auditor check strategy.py
python -m nba_auditor path strategies/ --json      # check_path
python -m nba_auditor backtest --code strategy.py --returns returns.csv --benchmark spy.csv --n-trials 40
python -m nba_auditor returns returns.csv --n-trials 40 --turnover 0.4 --json
python -m nba_auditor trades trades.csv --prices closes.csv
python -m nba_auditor explain LA07
python -m nba_auditor template vectorbt > strategy.py   # a backtest template: pandas, vectorbt or backtrader
python -m nba_auditor mcp          # the MCP server over stdio
```

**Input conventions**
- Returns are simple per-period decimals (0.01 = 1%). Use `units="percent"` for percent values, or `input_kind="equity"` for an equity curve.
- `freq` defaults to `auto` (inferred from the dates). A stated frequency that contradicts the dates is an error.
- Turnover is one-way: sum of |change in weights| per period. Cost in bps is charged per unit of turnover.
- Returns are treated as gross unless `returns_are_net=True`. When turnover is given without a cost, the headline result charges 10 bps.
- `n_trials` counts every variant you tried (parameters, universes, rules). By default the DSR assumes the trials' Sharpe ratios vary the way zero-skill strategies would (variance 1/T). If you know the actual variance, pass it as `trials_sr_variance` (annualized).


## References
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, JPM 40(5). Worked example reproduced in `tests/test_stats_paper.py`: DSR 0.9004; 0.9505 at N=46; 0.9505 at N=88 with Normal returns.
- Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier*, J. of Risk 15(2). MinTRL examples reproduced: 59.895 months; 2.73, 2.83 and 3.24 years.
- Lo (2002), *The Statistics of Sharpe Ratios*, FAJ 58(4). Newey & West (1987), Econometrica 55(3).
- See `explain(<id>)` for the references behind each check.

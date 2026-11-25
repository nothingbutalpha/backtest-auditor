# Known limitations

What the checks miss or treat specially, rule by rule, for version 0.7.0. This text is taken from the package README.
Changes from it: references to sections that were not carried over and to our internal records were replaced, early
figures without context were removed, and the summary below and the calibration table were added. The
checks miss many leaks and can raise false alarms on plain, correct files. Sealed test results from test set v6 on, including
the ones the Auditor failed, are published at https://nothingbutalpha.app/backtest-audit.

**In short.** The Auditor's static checks catch some common look-ahead leaks and miss many. In our latest full-size
sealed test (2026-10-18), they flagged 34 of 60 planted leaks. They can also flag correct code: in that test, 1 of
105 plain correct files and 4 of 55 correct files written to look like leaks (5 of 160 overall). On leaks that
open-source developers had found and fixed in their own code, they flagged 2 of 28 in a first trial, then 5 of 22
and 0 of 10 in two small pilots, which may repeat examples from the first trial (in the first pilot, possibly most
of them); with so few examples these counts are very uncertain. The full-size test and both pilots did not meet our
pass rule. Treat a clean result as weak evidence, never as proof that a backtest is free of leaks.

- **v0.5 loop order simulation: exactly what is exempt** (after red-team round 12). A condition in a bar loop over
  `i` is not reported only when all of these hold; otherwise it is judged as before (high).
  1. Every term is either past-only (no read of a later bar, no helper function defined in the file, no name tainted
     by a later bar's data other than bar `i+1`'s open) or a fill comparison: bar `i+1`'s low or high, exactly one bar
     ahead, read directly (`lo[i + 1]`, `df["low"].iloc[i + 1]`) or through a next-bar row alias (`nxt = bars.iloc[t + 1]`,
     also `.loc` and `rows[k + 1]`; not a single price copied into a variable, `nl = lo[i + 1]`), against a level known
     before that bar's high and low (past data, or bar `i+1`'s open).
  2. The fill comparison points the filling way: the low at or below the level (buy limit, sell stop) or the high at
     or above it (sell limit, take-profit, buy stop). The other direction (`nxt["low"] > level`) means the order did
     not fill: it is allowed only as a skip (`continue`, `break`, `pass`), never to book a trade (rt12 R06).
  3. The branch prices the fill or exit from the order's level (`min(limit, o[i + 1])`, `max(stop, o[i + 1])`,
     `exit_px = tp`, the limit itself). A price at a later bar's low or high itself (`c[i + 6] / lo[i + 1]`) is not
     exempt (rt12 R11). Later-bar reads in the branch must be prices assigned to fill, exit or trade-log names, plus the
     fill's timestamp (`bars.index[i + 1]`); a size, weight, volume or position from a later bar is not exempt.
  4. Fill and exit values (and names derived from them) reach only `append`/`extend`, a return value, or a
     print/log/file call; never a condition, a comparison, or another variable used that way, in this iteration, a
     later one or after the loop (rt12 R03 cooldown on the last trade's return, R04 keeping only winners). The list
     they are appended to inherits this (rt13): it may reach only summaries and sinks (`mean`, `len`, `sum`, `print`,
     a `DataFrame`/`Series`, a return value, a file, and names built from those). A read of it in an `if`/`while`
     test, a comparison, a comprehension filter, a loop over it, or a `break`/`continue` condition
     (`not trades or trades[-1] > 0`, `if sum(trades) < -0.2: break`, `[t for t in trades if t > -0.02]`) removes the
     exemption, and LA13 is reported high at that line. Reads of its size are not decisions (v0.6, the count-only trade
     cap): `len(trades)` anywhere, and truthiness tests (`if trades`, `not trades`, an `and`/`or` operand that is
     itself only a test), because how many fills have happened so far is known in real time.
  5. Decided from past data: an enclosing past-only condition, an `and` term that reads past bar data (`not in_pos`
     alone is state, not a decision), an `elif` of another fill check, or a loop over precomputed signal indices
     (`for i in idx`, not `range`) where nothing forward-looking feeds the indices, followed through names, function
     parameters to their call sites, object attributes (`self.idx = ...` in a class, `s.idx`; every assignment to
     that attribute name is checked, and an attribute with no visible assignment gives no exemption; rt13 X02),
     `np.flatnonzero`, `np.where(...)[0]`, `.nonzero()`, `.index`, comprehensions, and positional slices that start at
     different bars (`c[5:] / c[:-5]`; rt12 R01, R02). If the indices are tainted there is no exemption at all. An unconditional resting order on every bar (`limit = c[i] * 0.99` each bar, or a
     buy-stop at the 20-bar high, rt12 K01) counts as decided only when its fill is priced from its level.
  - When conditions 1-4 hold and the order was explicitly decided but the fill isn't priced from the level, the
    finding is "possible".
  - Deliberate (agreed with the red team in round 13): a level set from the fill bar's open (`limit = o[i + 1] * 0.99`,
    rt12 R10, directly or through names assigned earlier in the same branch) is exempt only when it sits on the far
    side of that open (below it for a buy, `o * k` with k < 1 or `o - x`; above it for a sell or take-profit, k > 1 or
    `o + x`) and the fill is booked at the level itself, not `min(limit, o[i + 1])`. Otherwise it is high. A stop derived from an entry at bar `i+1`'s open and checked inside an inner
    loop over later bars (a trailing stop, rt12 K02) is not reported by the outer loop.
  - Not exempt: event studies and order simulations that check bars beyond `i+1`.
- **Known bypass (backlog, to fix after v10):** any identifier or string containing `purge` (or `embargo`) turns off
  the v0.7 CV and walk-forward rules, and any `k - X` slice counts as a purge gap whatever `X` is (`PURGE = 0` is
  missed). This costs recall only.
- **v0.7 follows values through helpers, but only so far.** Helper return values, method calls on objects of a class
  defined in the file, `.pipe`/`.apply` targets, `self.<attr>` set in a constructor, list appends, loop targets and
  dicts of frames are followed; dynamic dispatch, imported helpers, `getattr`, decorators and values stored in
  containers indexed by computed keys are not. At most 25 call sites per helper are examined, and the flow search has
  a fixed work budget per file: on very large files a rule that runs out of budget reports nothing rather than guess.
  A statistic computed inside a helper is reported where it is written (the helper), not at the call that passes the
  full sample.
- **v0.6 rules lean on names and visible flow.** "Reaches a signal, position, PnL or model input" is decided by
  following assignments (and helper return values) to names the lint classifies as signals or PnL, or to model inputs.
  A transform that feeds a decision through names it can't classify stays "possible". The survivorship patterns only
  fire on price- or return-named wide frames, and the full-sample estimate rule (LA05) only when the file has no split,
  training window or walk-forward names and the positions are booked against returns. Two-sided transforms re-run on
  data up to each date inside a loop are treated as one-sided without checking the window.
- **Static analysis can't see everything.** Early versions caught only part of the leaky snippets our red team wrote
  for them (2026-10-16). Those snippets are now regression tests (`tests/corpus_rt2`), so any score on them is
  in-sample. A clean `check_code` result is weak evidence. Sealed test results, including the ones the Auditor
  failed, are at https://nothingbutalpha.app/backtest-audit.
  - Still missed (from an early red-team review and our own tests):
    - LA14: hyperparameter search with Optuna or `set_params` scored on the test set, and selection through a
      helper that returns the test Sharpe.
    - LA15: period ends renamed to `date` before the merge, `to_period` keys, period ends read from the index.
    - SB01: keeping names with a full count of dates (`dropna(axis=1)` on a wide price frame is caught since v0.6,
      a universe from `last_valid_index` since v0.7).
    - LA12: `pd.Grouper(freq=...)` bars joined back.
    - LA02: `iloc[::-1]` on a whole frame (often a fix for newest-first files, so it is not flagged).
    - LA09: `qcut`/`rank`/`scale` on a column with no visible time-series operation in the file (we can't tell a
      time series from one date's cross-section), `describe()` quantiles.
    - CO01: a cost basis used for something other than trading costs can hide the finding.
    - Limit fills at the day's low, resampled bars realigned in another file, and leaks hidden in other files or in
      the data itself.
- **v0.2 trades some recall for precision.** These are deliberate choices:
  - Full-sample statistics inside functions count only when the column is visibly a time series (built with
    `rolling`, `shift`, `pct_change` and similar in the same scope). Functions run per date (`groupby('date').apply`,
    loops over dates) are skipped. A cross-sectional scoring function that ranks one date's stocks is not flagged.
  - Same-bar *close* fills in loops are medium and only for entries. Exits at the close on a close-based stop are
    common and close to executable, so they are not flagged.
  - vectorbt's default fill at the close is still "possible"; only an explicit `price=close` is medium.
  - Anything we can't line up is "possible" at most: positions written through masks or variable offsets, engine
    settings passed through `**kwargs` (they may carry the fees), resample labels passed through `**kwargs`, and
    exact merges on a period end that are not forward-filled afterwards.
  - LA15 counts a reporting lag only when it is applied to the period-end column or index itself:
    `f['period_end'] + <offset>`, `f['period_end'].add(<offset>)`, `assign(period_end=... + <offset>)`,
    `f.index = f.index + <offset>` (or `+=`), or `f.shift(n, freq=...)` / `tshift(n)` with n > 0. An offset is a positive
    `pd.Timedelta`, `DateOffset`, `pd.offsets.*` (e.g. `QuarterEnd(1)`, `BDay(45)`) or `to_timedelta`, or a name bound
    to one. Not lags: `merge_asof(..., tolerance=...)` and other keyword arguments, zero offsets that only snap a date
    (`MonthEnd(0)`), negative or zero `shift(freq=...)`, and other index changes (`tz_localize`, `tz_convert`). Early stopping on a set named `holdout`/`oos` is not flagged when a separate `test` set is
    scored, and loops that redefine the test window each iteration are treated as walk-forward evaluation.
- **v0.3 trades more recall for precision** (LA05, LA10, CO01). Deliberate choices, each a possible miss:
  - LA01/LA10: a value computed from forward data under a neutral name (`up`, `oo`, `tmp`) is high if it reaches a
    signal or position, and medium if it reaches a model input. Model inputs are: the first argument of
    `fit`/`predict`, a feature-named assignment, the return value of a feature builder, a top-k pick (`nlargest`,
    `idxmax`), or a PnL product as a comparison. Used some other way, it is "possible" (v0.2: medium).
  - LA10: forward data used as the outcome is not an input: `groupby(...)[fwd].mean()` and other aggregations,
    `df.loc[mask, fwd]` / `df[mask][fwd]` with a mask built without forward data, and `position * fwd_ret`.
    Results built from these are "possible" when their name only suggests a leg or a spread (`long_leg`, `spread`,
    `trades`) and are not reported otherwise. An outcome is an input again, and high, when it is:
    - put back onto rows by `map`/`merge`/`reindex`/`transform`, a column assignment or a comparison;
    - assigned to a signal, position or weight name (`position`, `weight`, `score`; not names that only say long,
      short or trade);
    - a non-return factor (rolling statistic, sign, rank, clip, ratio) in a product with a return. Products are
      `*`, `@`, `.mul`/`.multiply`, `np.multiply`, `np.dot`, and `np.where(c, r, -r)` / `r.where(c, ...)`.

    Also high: a transform written inside the factor (`bench * np.sign(mkt)`,
    `np.sign(fwd.rolling(5).mean()) * fwd`), and a return series flipped or overwritten on rows chosen from forward
    data (`strat[tilt < 0] *= -1`, `strat.loc[tilt < 0] = -bench`).

    Missed:
    - an outcome read back some other way (`agg.loc[d]` inside a loop);
    - a product inside a helper function (`backtest(w, r)`, rt6 A06).

    Caught since v0.4 (were missed in v0.3): `pivot(values="fwd_ret")`, `np.copysign(bench, mkt)`,
    `frame.eval("t * b")` and `frame.query("m > 0")["b"]` under a strategy name.
  - v0.4 `query`/`eval`: only literal strings are read (`df.eval(expr)` with `expr` a variable is not), one statement
    per string. A frame built once from a dict literal maps its column names back to the source expressions; other
    frames read `name` as `frame["name"]`. A query is a row filter like `frame[cond]`, so an inline future-data
    condition is judged like `bench[mkt > 0]`: not counted unless the result is stored, unaggregated, under a strategy
    name (`strat`, `strategy`, `pnl`, `equity`, `backtest`, `bt`, `wealth`; not `perf`, `port` or `net`), which is medium.
  - v0.4 pivot: the `values=` column (keyword, or positional for `pivot_table`) is read. A `pivot_table` whose index or
    columns name buckets (`q`, `decile`, `quintile`, `bucket`...) is an outcome table, like `groupby(q)[fwd].mean()`.
  - v0.4 fill model (LA01/LA10): only the comparison inside the fill mask is exempt. The mask must be `<order decided
    without future data> & (<next bar's low or high, exactly one bar ahead> <op> <a price known at t>)`, and every use
    of the mask must select, zero or count trade returns (`r[filled]`, `r.where(filled)`, `np.where(filled, r, 0)`,
    `filled.sum()`). Any other use (a signal, a position, a feature, a fit, `astype`, a shift) makes it an input
    again, as before. Every other read of the next bar's low or high is still an input: sizing from
    `limit_px - low.shift(-1)` next to the fill mask is LA10 high (rt10 G2), and an entry or exit priced at
    `low.shift(-1)` / `high.shift(-1)` is LA01 high (rt10 G7), also when the price is written inline in the trade
    return selected by the mask. A column read from a frame shifted as a whole (`best = df[["low", "high"]].shift(-1)`,
    then `best["high"]`, `best.loc[:, "high"]` or `best.high`) is read exactly like that column shifted by -k, for the
    price rule, the sizing rule and the mask exemption alike (rt11 G9). A two-bar-ahead low, a fill flag without an
    order term, or a fill flag with a fill-sounding name (`will_fill`) is still flagged. Missed: a numpy fill model
    written with offset slices (`lo[1:] <= lim[:-1]`, sizing from `lo[1:]`, rt11 G13).
  - v0.4 LA13: a slice whose start is in the past and end in the future (`x[i - 2: i + 3]`) is not read as a
    future read (unchanged from v0.3). The label-only exemption follows each loop output to its uses: returned by a
    function (optionally wrapped in `pd.Series(...)`) whose every call site assigns a label-named target, one passed
    as a label, or any non-signal target when the function's name says label or target; and from there only into
    `fit`'s y, `label=`, a score, a label- or evaluation-named name, a summary (`value_counts`, `describe`, `mean`,
    `groupby(...).mean()`, `plot`) or a file (`to_csv`, `to_parquet`), with no product or comparison on the way.
    Index-only reads (`X.loc[y.index]`, `reindex(y.index)`, `len(y)`, `y.shape`) are alignment and don't count.
    Since red-team round 10, a label written as a column (`px["target"] = labeler(...)`) is followed through its
    column uses exactly like a name, and a bare call statement is a sink only when it is a print, plot or file write:
    `positions.extend(labels)`, `.append`, `.insert`, `.update` and `df.insert(1, "lab", labels)` are writes into the
    other name or column, which is then followed in turn (rt10 L1, L2, L3; L5, a `where` on the labels, is caught too).
  - v0.4 performance guards: call sites, name reads, the statements that read each name or column, label uses and
    the "feeds a signal" graph are each indexed once per file, and the open-fill and fill-mask rules skip files or
    values that can't match. The red team's worst cases (rt10: 1,100 labeler functions at 232 KB, 1,600 label loops at
    247 KB, 1,600 fill models at 268 KB) run in about 2.5 s on our development laptop, and their 262 KB symbolic/container
    stress file in 2.3 s, the same as main (main: 2.2-2.6 s for the first two; the red team measured 165 s on main
    for the fill models, which hit a quadratic scan that predates v0.4). `tests/test_redteam_rt10.py` times each shape against the 3 s local budget and the 8 s ceiling.
  - Still missed after round 10 (documented, `KNOWN_MISSES` in `tests/test_redteam_rt10.py`):
    - H2: a loop with no condition that writes forward values (`label_max[i] = c[i + 1: i + 21].max() / c[i] - 1`)
      under a label name, then used as a feature. It isn't LA13's pattern, and the loop's output isn't tracked as
      forward data.
    - H5: a helper loop that writes forward values, smoothed into a signal outside the helper.
    - J5: forward columns merged in from a second frame (`px.merge(labels, on=[...])`), then listed as features.
    - Q1: `strategy = df.query("fwd_ret > 0")` followed only by a mean, which reads like a capture statistic. Anything else (a feature, a
    signal, a product with a return) keeps the high. Event studies that branch on post-event values are flagged.
  - v0.4 LA09 in `apply`: the receiver must be visibly a time series (built with rolling, shift, pct_change and similar
    here or where its base frame was assigned) and feed a signal, a feature or a model for medium; otherwise
    possible.
  - LA10 containers: a frame built from columns that stores a forward column is not reported where it is built.
    It is reported (high) where a forward column survives into `fit`/`predict`, a training dataset, a feature matrix
    or a signal.
    - Names come from dict keys, `.rename("x")` and `keys=[...]`. They are kept only through two kinds of change:
      - operations known to keep names: row filters, `dropna`, `fillna`, sorting, index changes, `astype`, `copy`,
        `to_numpy`, `assign`, and a `drop` that is resolved by name;
      - renames with literal names: `rename(columns={"a": "b"})` (also with `inplace=True`) and
        `data.columns = ["a", "b"]` while the column order is known.
    - Any other frame-level operation on a container makes its column names unknown. This includes in-place calls,
      `add_prefix`, `set_axis`, `rename(mapper, axis=1)`, `rename(columns=lambda ...)` and
      `.columns = .columns.str.replace(...)`.
    - A forward column with an unknown name survives every later explicit column list or drop.
    - Positional selections (`iloc[:, :-1]`, `arr[:, -1]`) are resolved while the column order is known. The order
      is updated or dropped as columns move:
      - a literal column list (`data[["c", "a", "b"]]`, `.loc[:, [...]]`, `reindex(columns=[...])`) sets the order
        to that list;
      - a resolvable positional selection (`iloc[:, ::-1]`) reorders it;
      - any other column-axis change makes it unknown: `sort_index(axis=1)`, a transpose (`.T`), a column subscript
        that isn't a literal list (`data[sorted(data.columns)]`, `data[data.columns[::-1]]`), a non-literal
        reindex, or any operation the lint can't read.
      With the order unknown, a later positional selection keeps the forward column.
  - LA10 masks: a return selected by a mask built from future data (`bench[long]`, `bench.loc[short]`, `bench[~m]`,
    also through `.values`, `.to_numpy()` and `m[m].index`) is high when the mask has a position name and medium
    when it has a neutral one (`ok`, `up`). An inline comparison (`bench[mkt > 0]`) is not counted, so up/down
    capture statistics stay clean; a neutral-named capture mask gets the medium.
    - A row shift of the whole frame (`data.shift(5)`) lags every stored column. A forward column whose lead is
      fully cancelled (a 5-bar lead, `shift(5)`) no longer counts; a partial lag (`shift(1)`) still does.
  - Look-ahead cancel scope: a lead and a lag written with the same expression (`shift(-h)` then `shift(h)`) cancel
    only when every name in it is bound once and is not a loop variable. A module constant shadowed by a function
    parameter or local is not used. Missed: a helper defined above the line that creates the forward column it reads
    (the pass runs in source order).
  - LA10: a model trained on the label (`fit(X, y)`, `lgb.Dataset(X, label=y)`, `xgb.DMatrix`, `Pool`) no longer
    passes "future data" on to the model and its outputs. Whether the training rows overlap the evaluated rows, or
    end less than a label horizon before them, is LA05/LA14's job, and fits inside functions are not checked there.
  - Look-ahead size: constants (`H = 5`, `cfg.horizon`) and simple arithmetic (`-H`, `H + 1`, `-1 * 5`) are
    resolved. Any other expression cancels only the same expression (`shift(-horizon)` then `shift(horizon)`). A
    different unknown lag counts as 1 bar.
  - LA10 column lists: drop lists and feature lists are resolved per column (string lists, `+`, single-assignment
    constants, comprehensions over `df.columns` with `startswith`/`endswith`/`in`/`==`/`re.match`, and
    `df.columns.str.startswith`/`isin` masks). A list the lint can't resolve (built in a loop, read from a file,
    reassigned) gives "possible" for the forward columns it may keep, never high.
  - LA05: a slice or date mask bounded above (`.loc[:cut]`, `.iloc[:k]`, `[:k]`, `.loc[a:b]`, `df.date < cut`,
    `between`) counts as a training subset, as does a frame assigned once from one. A start filter (`.loc['2000':]`,
    `[20:]`, `df.index >= '2000'`) or a test-window mask does not. The lint does not check that the evaluation uses
    the later rows. Missed as a result:
    - an in-sample score on the same slice;
    - fits on all rows but the last (`.iloc[:-1]`), which count as subsets;
    - overlapping train and test slices (`iloc[:3000]` vs `iloc[2500:]`);
    - a model fitted and scored on the same `iloc[:-1]` rows.
  - Also still missed from the round 5 review: a forward-filtered ticker list used through `isin`, an
    `inplace=True` drop on a different frame than the one fitted, and a forward column re-joined under another name
    after a drop.
  - CO01: never above "possible" (3 points off the score). Missing costs are a realism caveat, not a lookahead
    error, and the returns may already be net.
- **Suggested fixes are narrow and change behavior on purpose.** `shift(-1)` becomes `shift(1)`, which uses older
  data than the original author may want; `TimeSeriesSplit` gives different folds than `KFold`. The self-check only
  shows that the lint no longer sees the pattern and sees nothing new; it doesn't run the code. Fixes aren't offered
  for notebooks yet. The agent-fix eval is in-sample.
- **`check_path` resolves relative paths against the server's working directory**, which depends on the client;
  agents should pass absolute paths. It has **no scope limit**: it reads any local `.py`/`.ipynb` file or folder the
  user running the server can read (it never writes). Network (UNC) and device paths are refused, for `check_path`
  and for every data-file argument.
- **Results quote file content.** Paths, snippets and suggested edits come from the files and are marked
  `content_is_untrusted_data`; a file can contain text written to look like instructions. Snippets have comments
  stripped. Data files (`.csv`, `.json`, `.txt` only) are never quoted in error messages.
- **The name heuristics can be fooled.** A label called `signal` will be misread. Test sets are recognized by
  name (`X_test`, `test`, `oos`, `holdout`), validation sets by `val`/`valid`, and fundamentals dates by name.
- **It reads Python only.** Pine Script and other languages are not supported yet.
- **The verdict depends on `n_trials` being honest.** In the calibration below, the best of 100 noise strategies
  reported with `n_trials=1` got "No red flags found" in 67% of 200 runs.
- **Trade-log survivorship checks are heuristics, not proof.** They flag symbols traded before the current
  company used them (some symbols were reused, e.g. ARM, SNOW, CEG, META) and universes where no name ever
  disappears. The built-in list of first-trade dates is short.
- **Trade logs without daily prices are only marked when positions trade.** Their Sharpe is overstated, so the
  verdict can never be "No red flags found".
- **The cost model is linear in turnover.** No market impact, borrow fees or financing.
- **Intraday series are reviewed but never get "No red flags found".** The implausibility thresholds assume daily or
  slower bars.
- **Verdict labels were renamed** from "Likely real / Inconclusive / Likely artifact" to "No red flags found / Some
  red flags / Major red flags", because "Likely real" implied an edge. A compliance-checker pass on 2026-10-16
  edited the disclaimer and several explain texts; public copy, including this repository, gets its own pass.
- **Input limits.** `check_code` accepts up to 1,000,000 characters; MCP data arguments up to 20,000,000
  characters each. There is no wall-clock timeout; these limits keep a run to seconds on a laptop.

## Calibration of the return-series verdict on synthetic data (how often the verdict is wrong)

These are synthetic daily series of 5 years each (1260 days; the leveraged-index rows use 10 years), with
turnover 0.05 per day. Every row gets an independent market series as its benchmark, except where it says
"no benchmark". Without a benchmark the best verdict is "Some red flags" by design. Reproduce the table with
`python scripts/calibration.py` (fixed seeds; re-run on version 0.7.0 on 2026-11-22 with the same figures).

| Series | No red flags found | Some red flags | Major red flags |
|---|---|---|---|
| Pure noise, one strategy, `n_trials=1` (300 runs) | **1%** | 45% | 54% |
| Best of 100 noise strategies, honest `n_trials=100` (200 runs) | **0%** | 28% | 72% |
| Best of 100 noise strategies, **`n_trials=1` (understated)** (200 runs) | **67%** | 33% | 0% |
| Noise as 20-day overlapping returns, `n_trials=10` (200 runs) | **0%** | 6% | 94% |
| Noise with AR(1) autocorrelation 0.3, `n_trials=10` (200 runs) | **0%** | 4% | 96% |
| 2x leveraged index (Sharpe ~0.75), **no benchmark** (200 runs) | **0%** | 94% | 6% |
| 2x leveraged index, benchmark = the index (200 runs) | **0%** | 94% | 6% |
| Planted Sharpe 0.5, `n_trials=10` (200 runs) | 1% | 22% | 76% |
| Planted Sharpe 1.0, `n_trials=10` (200 runs) | 14% | 54% | 32% |
| Planted Sharpe 1.5, `n_trials=10` (200 runs) | 48% | 44% | 8% |
| Planted Sharpe 0.7, positive skew, `n_trials=1` (200 runs) | 0% | 100% | 0% |
| Planted Sharpe 0.7, negative skew, `n_trials=1` (200 runs) | 0% | 100% | 0% |

# Lint recall analysis for v0.7 (dev sets v4-v9)

> **Development record.** These figures were measured while building the lint. Most are in-sample: they come from
> development sets whose misses we had already read, or from test files we wrote. None is an accuracy measure, and
> they overstate what to expect on new code. Sealed test results, including the ones the Auditor failed, are at
> https://nothingbutalpha.app/backtest-audit. The checks miss many leaks and can raise false alarms on plain,
> correct files.

Written 2026-10-18 at the start of the v0.7 recall sprint, from the lint at main 57879a8 (lint v0.6, lint.py sha256
2fe76fcf..., the commit scored on sealed v9). The sealed v9 run is done and recorded in
our internal sealed-run record, so v9 is a development set from here on, like v2-v8. Every number below and
every number this sprint produces on v2-v9, on the twins or on the red-team corpora rt14-rt16 is in-sample. The next
public number needs a sealed v10.

## Baseline (lint v0.6, scored with each set's own scorer: a leak is caught only by a high/medium finding within
+/-2 lines of a labelled leak line; a correct file with any high/medium finding is a false alarm)

| Set | Recall | False alarms on correct files (plain / near-miss) |
|---|---|---|
| v2 | 37/50 | 0/30 |
| v3 | 41/60 | 0/40 |
| v4 | 30/60 | 0/50 (0/20, 0/30) |
| v5 | 33/50 | 0/70 (0/40, 0/30) |
| v6 | 36/60 | 0/80 (0/50, 0/30) |
| v7 | 35/60 | 0/90 (0/55, 0/35) |
| v8 | 35/60 | 3/160 (0/105, 3/55: clean_17, clean_27, clean_141) |
| v9 | 30/60 | 4/160 (0/105, 4/55: clean_043, clean_048, clean_086, clean_115) |
| rt14 | 19/20 | 2/22 (q10 LA10, q18 LA05, both pre-v0.6) |
| rt15 | 12/12 | 1/9 (n04 LA10, pre-v0.6) |
| rt16 | 4/8 | 0/8 |

v9 by difficulty: easy 12/15, medium 12/25, hard 6/20. By category: selection-bias 0/3, survivorship 2/5,
lookahead 12/25, fill-timing 3/6, target-leakage 2/4, normalization-leakage 4/7, cv-leakage 4/6, data-snooping 3/4.

151 leaky files are missed: 30 on v9 and 121 on v4-v8 that v0.6 still misses (v4 30, v5 17, v6 24, v7 25, v8 25).
Each was read and assigned to one class by the mechanism of the leak (not the set's category label). A file that fits
two classes is counted once, under the first. The v4-v8 files keep the v0.6 analysis' reading; the classes are
regrouped around what is still open.

## Miss classes, ranked by frequency x feasibility

Feasibility: **high** = a narrow structural pattern with few legitimate look-alikes; **medium** = needs a flow
condition (the value reaches a signal, position, model or PnL, and no split or window is visible) to stay precise;
**low** = needs knowledge the code doesn't show (timestamps, vendor semantics, event windows, corporate actions).

| # | Class | v9 | v4-v8 | Feasibility | Files |
|---|---|---|---|---|---|
| A | **Interprocedural flow**: a full-sample statistic or fit written inside a helper, a `.pipe`/`.apply` function or a class constructor, and used at a call site on the full frame (z-scores, winsorizing, scalers and imputers, vol targeting with the full-sample std, `polyfit` detrending, mean-variance weights, two-sided filters, reversed expanding windows, per-ticker `groupby.apply` z-scores, `SelectKBest` in a helper); also future-derived series passed through helpers and returns (a negative-lag helper call, `filter(like=...)` catching a forward column built in a helper, a reversed diff in a helper, a centred slice in a class method) | 6 | 18 | medium | v9/06,09,18,37,49,53; v4/23,51,54,55,56; v5/14,26,42,46; v6/16,37,42,55; v7/01,16,58; v8/53,58 |
| B | **Survivorship**: `dropna(axis=1)` reached through `.loc[...]`, `.pipe(lambda d: ...)` or `resample` before the returns; `cols[px.iloc[-1].notna()]`; a universe cut from the last row by a threshold (`last[last > x].index`); current members (`end_date.isna()`) inside a helper or a `query("...isnull()")` string; a "current"/"latest" constituents file; a live API list; members as of the end date; `nanmean` over names that delist | 3 | 13 | high for the first four, medium for files and helpers, low for live APIs and end-date members | v9/02,40,48; v4/28,49,50; v5/30; v6/49; v7/28,37,38,48,55; v8/04,09,41 |
| C | **Selection on the full sample**: features chosen by `corrwith` / `SelectKBest` / mutual information on all rows and then cross-validated; pairs chosen by full-sample cointegration and backtested on the same rows; min-variance weights from the full-sample covariance; ARIMA/HMM fitted or decoded on all rows | 2 | 6 | medium (needs: no split before the selection, the selection reaches the CV, model or PnL) | v9/15,58; v4/43; v6/13,15; v7/24; v8/21,31 |
| D | **Finding on the wrong line**: forward column in a feature-list constant more than 2 lines above the fit; the test-scoring call inside a selection loop; rolling means of the label reported at the fit; `right_on="period_end"` on its own line of a `merge_asof` (LA15 lands on the call line) | 1 | 10 | high | v9/16; v4/01,09; v5/08,09; v7/02,03,22,23; v8/01,35 |
| E | **CV and walk-forward**: `cross_val_score(cv=<int>)` (unshuffled K-fold) with a multi-bar forward label; a hand-written walk-forward loop without a purge gap for a multi-bar label; `TimeSeriesSplit` without `gap` that the v0.6 rule misses (horizon held in a module constant); a grid search fitted on all rows and then scored on a later slice; shuffled `StratifiedKFold`; `random_split` over overlapping windows | 2 | 7 | high | v9/25,54; v6/28,29,59; v7/13,45; v8/33,43 |
| F | **Point patterns with a structural tell**: `groupby(level="ticker").rank(pct=True)`; per-day aggregates keyed by a day *column* (`df["day"] = df.index.date`, then `groupby("day")...transform("sum")`); pooled `qcut`/`quantile` across all dates of a panel; `percentileofscore` against the full history; full-history `rank(pct=True)` that stays "possible"; limit checked against the same bar's low in an `iterrows` loop; statsmodels `smoothed_state`; an all-time high (`px.max()`) driving a drawdown signal; fills open-to-close on a close signal; test-set pickers over a dict comprehension | 8 | 21 | mixed (the first eight high or medium; daily joins and open-to-close fills low) | v9/04,13,34,35,41,43,50,60; v4/24,27; v5/01,04,22,41,50; v6/09,10,18,35,39,40; v7/04,09,19; v8/07,28,36,39,45 |
| G | **Not attempted** (needs information the code doesn't carry, or would trade precision): interpolation (possible by policy), point-in-time stamps and releases, after-close news, time zones, corporate actions, target encoding and meta-label features, event windows, dollar-bar thresholds, intrabar order of stop and target, trailing-stop ratchets, window slices that include the target, cost omission (possible by policy), other | 8 | 46 | low | v9/19,23,24,42,46,47,51,56; v4 classes 10/11/13-17 and v4/19,48,60; v6/30,54; v7/17; the rest of v0.6's classes 10, 11, 13-17 |

Totals: A 24, B 16, C 8, D 11, E 9, F 29, G 54 (151).

## Priorities for v0.7

The brief names three: interprocedural dataflow, selection bias and survivorship. They are also the three largest
classes with at least medium feasibility after the point patterns, and v9's two worst categories (selection-bias 0/3,
survivorship 2/5) sit in B and C. Order of work: D (cheap, precise), B, A, C, E, then F. G is not attempted.

The pre-registration of what is attempted, how it is measured and the builder's own twins is in
`tests/corpus_v07/PREREGISTRATION.md`, committed with this file and before any change to `lint.py`.

## Result (added after the build; in-sample)

Every number below is in-sample: the rules were written after reading these misses. The honest test is a sealed v10.

| Set | Recall, v0.6 → v0.7 | False alarms, v0.6 → v0.7 (plain / near-miss) |
|---|---|---|
| v2 | 37 → 37/50 | 0/30 → 0/30 |
| v3 | 41 → 44/60 | 0/40 → 0/40 |
| v4 | 30 → 39/60 | 0/50 → 0/50 (0/20, 0/30) |
| v5 | 33 → 41/50 | 0/70 → 0/70 (0/40, 0/30) |
| v6 | 36 → 44/60 | 0/80 → 0/80 (0/50, 0/30) |
| v7 | 35 → 50/60 | 0/90 → 0/90 (0/55, 0/35) |
| v8 | 35 → 46/60 | 3/160 → 3/160 (0/105, 3/55) |
| v9 | 30 → 49/60 | 4/160 → 4/160 (0/105, 4/55) |
| rt14 | 19 → 20/20 | 2/22 → 2/22 |
| rt15 | 12 → 12/12 | 1/9 → 1/9 |
| rt16 | 4 → 4/8 | 0/8 → 0/8 |

- No leak caught by v0.6 is lost on any set, and no correct file (plain or near-miss) gained a high or medium finding.
  The pre-existing false alarms (v8 clean_17, clean_27, clean_141; v9 clean_043, clean_048, clean_086, clean_115;
  rt14 q10, q18; rt15 n04) are unchanged.
- v9 by difficulty: easy 12 → 14/15, medium 12 → 21/25, hard 6 → 14/20. By category: selection-bias 0 → 3/3,
  survivorship 2 → 4/5, cv-leakage 4 → 6/6, normalization-leakage 4 → 6/7, lookahead 12 → 21/25, fill-timing 3 → 4/6,
  data-snooping 3/4, target-leakage 2/4.
- Pooled v4-v9: 199 → 269/350 (easy 66 → 77/87, medium 82 → 113/145, hard 51 → 79/118).
- Twins: leaky 2/43 → 43/43 flagged as expected; clean 0/41 with a high or medium finding, before and after.
- Catches that count only because of a "same issue as line N" repeat: 7 (v0.6) → 18 (v0.7) across v2-v9. On v9: 2
  (leaky_12, leaky_16); without them v9 is 47/60.
- Post-hoc (not in the pre-registration): the two template-build gaps (vectorbt `price=open` with unshifted entries; a
  helper-returned signal traded at the same bar's open) and five rule extensions P1-P3, P5, P6 (see
  `tests/corpus_v07/posthoc/README.md`). P6, the k-fold CV rule firing on 1-bar labels (pre-registered: more than one
  bar), was recorded after the red-team review (F1) and kept deliberately. Without them the dev sets are v3 43, v4 38,
  v5 40, v6 42, v7 48, v8 45 (v2, v9, rt14-16 unchanged): 9 dev catches come from post-hoc rules (v3 leaky_21 depends
  on P6).
- Fresh probes (red team, rt17, 56 files written after v0.7 was frozen at 70344f5): leaks 3/17 → 6/17, false alarms
  on correct files 4/39 → 5/39 (the added one, q17, is P6). The new rules catch the twins' syntax but miss plain
  rephrasings of the same mechanics, so the dev-set gains overstate what to expect on a sealed v10: expect a much
  smaller gain there.
- Still missed, by class (81 of 151): A 5 (v9 leaky_09 and leaky_37 are flagged at the statistic inside the helper, as
  pre-registered, while their label is the call line; `filter(like=...)` catching a forward column built in a helper; a
  centred slice in a class method; an hourly bar joined as-of through helpers); B 6 (current lists from live APIs, a
  file stamped with the end date, members as of the end date, current members with add dates in a helper, `nanmean`
  over names that delist); C 2 (an HMM decoded on all rows, ARIMA fitted on all rows then scored after a split date);
  D 0; E 1 (`random_split` over windows in a file with no time-series hint); F 13 (daily joins onto minutes,
  open-to-close fills on a close signal, a vectorized limit fill flag, ticker-year z-scores, a tensor normalized in a
  class method, a test-set picker over a sliced Sharpe, pooled `qcut` in an f-string loop); G 54 (not attempted).
- Real code (rescanned against v0.6): our internal research code and an earlier internal
  codebase, 948 files. Added high/medium: 1 true positive (risk-parity weights from the full-sample covariance of a
  strategy-return matrix, applied to that same matrix with an in-sample Sharpe printed) and 2 LA15 repeats on the
  `right_on="period_end"` line of a merge v0.6 already flags. Four kinds of false alarm seen during the build were
  fixed before this rescan (forward outcome returns written as a ratio of shifts, calendar `dates.shift(-1)`, metric
  helpers such as R² or volatility of a return series, a date column's `min()`, `dropna(axis=1)` on a window that
  moves with the rebalance date).
- Process record: the analysis, pre-registration and twins (commit "recall analysis, pre-registration and dev twins")
  and the template-build post-hoc twins were committed before any `lint.py` change; the rule for those gaps came next, then the rest
  of the rules in one commit. The sprint was interrupted once and resumed by a second builder session, which rebased
  the branch onto main, re-ran every number above from scratch (same results), re-ran the real-code rescan (950 files
  now, the 2 new files have no findings; the added findings are the 3 listed above) and removed one dead clause.
- The two `test_known_gap_*` strict xfails in `tests/test_draft_backtest.py` now pass, so they are plain asserts. The
  draft_backtest prompt's step 4 named both as gaps until the 0.7.0 release, which removed them from the prompt
  (it still asks the assistant to confirm the one-bar shift by hand).
- Speed: on a quiet machine, the 200 KB corpus blob takes 2.1-2.4 s (v0.6: 1.8-2.7 s in the same interleaved runs) and
  the rt10 shapes 2.8-3.1 s (v0.6: 2.4-3.7 s); every speed test passes. v0.7's rules add about 0.4 s per 200 KB.

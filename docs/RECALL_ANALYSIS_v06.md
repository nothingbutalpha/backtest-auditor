# Lint recall analysis for v0.6 (dev sets v4-v8)

> **Development record.** These figures were measured while building the lint. Most are in-sample: they come from
> development sets whose misses we had already read, or from test files we wrote. None is an accuracy measure, and
> they overstate what to expect on new code. Sealed test results, including the ones the Auditor failed, are at
> https://nothingbutalpha.app/backtest-audit. The checks miss many leaks and can raise false alarms on plain,
> correct files.

Written 2026-10-17 at the start of the v0.6 recall sprint, from the lint at main a8d57fc+ (package 0.5.0). v2-v8 are
development sets now (their scores are in-sample from here on); v9 will be written later and stays sealed.

## Baseline (v0.5.0, scored with each set's rules: a leak is caught only by a high/medium finding within +/-2 lines of a
labelled leak line; a correct file with any high/medium finding is a false alarm)

| Set | Recall | False alarms on correct files |
|---|---|---|
| v2 | 37/50 | 0/30 |
| v3 | 34/60 | 0/40 |
| v4 | 14/60 | 0/50 |
| v5 | 21/50 | 0/70 |
| v6 | 22/60 | 0/80 |
| v7 | 18/60 | 0/90 |
| v8 | 19/60 | 4/160 (1/105 plain: clean_16, LA12 medium; 3/55 near-miss) |

196 leaky files in v4-v8 are missed. Each was read and assigned to one class below (by the mechanism of the leak,
not by the set's category label). Counts are files; a few files fit two classes and are counted once, under the first.

## Miss classes, ranked by frequency x feasibility

Feasibility: **high** = a narrow structural pattern with few legitimate look-alikes; **medium** = needs a flow
condition (the result reaches a signal, position, model or PnL) to stay precise; **low** = needs knowledge the code
doesn't show (timestamps, corporate actions, data vendor semantics).

| # | Class | Files | Feasibility | Examples (set/file) |
|---|---|---|---|---|
| 1 | **Two-sided (non-causal) transforms**: `filtfilt`, `hpfilter`, `savgol_filter`, `gaussian_filter1d`, `np.gradient`, `argrelextrema`, `np.convolve(mode="same")`, Kalman `.smooth`, `seasonal_decompose`, `rolling(center=cfg.center)` from a config default | 19 | high | v4/37,38,42,44,45,46; v5/44; v6/27,43,45,47; v7/15,41,47,51; v8/20,25,37,40 |
| 2 | **Survivorship patterns**: `dropna(axis=1)` (also `thresh=len(d)`) on a wide price frame; columns kept by `iloc[-1].notna()`; universe from the last row (`iloc[-1].nlargest(n)`, latest market cap); winners by full-period return; current members (`removed.isna()`, `end_date.isna()`, "current" constituent files, members as of the end date); `groupby(id).filter(max date >= END)` | 21 | medium-high | v4/28,29,49; v5/23,27,33; v6/06,07,24,49,57; v7/27,28,37,38,48,55,60; v8/04,08,09,41 |
| 3 | **Full-sample normalization variants**: per-asset full-history `groupby(asset).rank(pct=True)`; winsorization at full-sample quantiles and `fillna(frame.mean())` before a date split; z-scores in helpers, classes or tensors; scaler/imputer/selector fitted inside a helper before CV; vol targeting with full-sample std; reversed expanding windows | 22 | mixed (per-asset rank and pre-split winsorize/impute high; the rest medium-low) | v4/23,51,55,56; v5/26,38,41,42,46,50; v6/09,10,40,55,60; v7/19,30,34,58; v8/36,53,54,58 |
| 4 | **Full-sample estimation driving the backtest**: PCA fit on all rows, full-sample OLS/`polyfit` hedge ratio, cointegration-selected pairs, `KMeans.fit_predict`, ARIMA/GARCH fit on all rows, full-sample covariance weights, HMM decoding over the whole sample | 16 | medium (needs: no split, output reaches positions/PnL) | v4/22,43,57; v5/28,47; v6/13,14,15,41,46; v7/24,29; v8/02,19,21,26,31 |
| 5 | **Finding on the wrong line** (the leak is seen, the finding lands > 2 lines away): forward column in a feature-list constant (reported at the fit); test scoring inside a selection loop (reported at the loop); `max(results, key=results.get)` (reported at the loop); `merge_asof` `direction=`/`right_on=` keyword lines; a negative-shift helper (reported at the call) | 14 | high | v4/01,09,36; v5/08,09; v7/02,08,22,25; v8/01,06,35,42,47 |
| 6 | **Cross-validation and sampling leaks**: `TimeSeriesSplit` without `gap` on a multi-bar label; SMOTE/`fit_resample` before the split; `.sample(frac=1)` or torch `random_split` on time-ordered rows; shuffled K-fold where the file isn't recognized as time series; a grid search fitted on all rows, then scored on a slice | 14 | high | v4/52,54; v5/03,17; v6/28,29,59; v7/13,16,39,45; v8/17,33,43 |
| 7 | **Model or parameter chosen on the test set by a picker**: `max(grid, key=lambda t: f1_score(y_test, ...))`; a dict of test scores then `max(d, key=d.get)`; an Optuna objective that returns a test-set score | 10 | high | v4/25,53; v5/04,05,48; v6/26,35; v7/54,56 |
| 8 | **Fills at the bar's own low/high/VWAP**: entry priced at the same bar's low; `from_signals(price=low)`; a same-bar limit checked against that bar's low; execution at the same day's VWAP after a close signal; entries selected by a close signal and booked open-to-close | 9 | medium-high | v4/21; v5/36; v6/18,39; v7/09,11,35; v8/07,24 |
| 9 | **Intraday period aggregates broadcast to earlier bars**: `groupby(<day/week/month of the index>)[col].transform("last"/"max"/"sum"/"std")`, `pd.Grouper(freq=...)` transform; joins of daily aggregates onto minutes | 10 | high for transforms (6), low for joins (4) | v4/26,27; v5/01,22; v6/30,34; v7/31; v8/32,39,45 |
| 10 | **Point-in-time joins**: period end renamed to `date`; quarter-end + `MonthEnd(0/1)`; `merge_asof` direction from a config dict; weekly bars relabelled earlier | 6 | medium | v4/08,47; v5/19,34; v6/30; v7/42 |
| 11 | **Same-bar products through wrappers**: `pd.Series(np_signal, index=...) * ret`; vol-target leverage from an unlagged rolling std times the same return; month-end weights forward-filled without a lag; weights times returns inside a helper | 6 | medium | v4/02; v5/43; v6/08,11; v7/06; v8/60 |
| 12 | **Other look-ahead with a structural tell**: backtrader `self.data.close[1]`; a lag helper called with a negative lag; a negative shift inside a groupby-transform lambda (currently "possible"); a centered slice window in a class method | 5 | high | v4/41; v6/16; v7/01; v8/28,44 |
| 13 | **Interpolation** (`interpolate(method="time")`) used before returns (currently "possible") | 4 | medium | v4/40; v5/25; v6/21; v8/15 |
| 14 | **Target encoding / label-derived features**: `groupby(g)["y"].transform("mean")` as a feature; same-day label reconstructed from features; meta-label features with realized barrier returns | 6 | low-medium | v4/17; v6/48; v7/14,21; v8/05,13 |
| 15 | **Calendar, time zone and news timing** (after-close events stamped the same day, other-market closes) | 7 | low | v4/15,33,34,59; v5/32; v6/11 (counted under 11); v7/05 |
| 16 | **Corporate actions** (unadjusted prices, back-adjusted price filters, today's share count) | 4 | low | v4/07,31,32; v8/49 |
| 17 | Other (cost omission reported "possible" by policy, newest-first API data, inclusive `.loc` split, event-window betas, factor-zoo snooping, window slice that includes the target) | 9 | low | v4/06,30; v6/03,56; v8/23,50,51,52; v7/07 |

## False alarms on v8 (to fix)

- clean_16 (plain): two series resampled with the same rule (`sec`, `mid`), then `feat.join(target)` where `target` is
  derived from `mid`. LA12 medium twice. The join is between bars with the same stamps; the lint only knows names
  assigned directly from `resample`, not names derived from them.
- Near-miss (should also go): clean_17 (a fill flag `nxt_low <= limit` with no order term, used only in
  `.where(filled)`), clean_27 (a numpy lag helper `out[k:] = x[:-k]`), clean_141 (label written through masks).

## Plan for v0.6

Implement, precision first, in this order: classes 1, 5, 7, 6, 2, 9, 8, 12, 3 (per-asset rank, pre-split winsorize and
impute only), 4 (only when there is no split and the estimate reaches positions or PnL), 13; fix clean_16 and the
count-only trade cap (Y04: `if len(trades) >= 50: break` reads only how many fills happened, which is known in real
time). Not attempted: classes 10, 11, 14-17 (they need information the code doesn't carry, or would trade precision).
Each implemented class gets the builder's own clean and leaky twins, committed with a pre-registration before the fix
(`tests/corpus_v06`). Dev-set recall before and after is reported per set, and labelled in-sample.

## Result (added after the build; in-sample)

Implemented classes 1, 2, 5, 6, 7, 8, 9 (transforms only), 12, 3 (per-asset rank, pre-split winsorize and impute) and
4 (no split, positions booked against returns), plus the clean_16 and Y04 fixes. Class 13 (interpolate) was left at
"possible". Dev-set recall, v0.5.0 to v0.6 (the rules were designed after reading these misses, so this is the most
in-sample number we have): v2 37 to 37/50, v3 34 to 41/60, v4 14 to 30/60, v5 21 to 33/50, v6 22 to 36/60, v7 18 to
35/60, v8 19 to 35/60. No leak caught by v0.5.0 was lost; false alarms on correct files: v2-v7 0, v8 4 to 3 (clean_16
fixed; the near-miss files clean_17, clean_27 and clean_141 remain). Details in docs/RULES.md and docs/LIMITATIONS.md. The honest
test is the sealed v9 run.

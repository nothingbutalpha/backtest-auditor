# Lint v0.6 (recall sprint): pre-registration

Committed before any change to `lint.py` and before v0.5.0 (main) was run on these twins. The miss classes come from
`auditor/docs/RECALL_ANALYSIS_v06.md` (v4-v8 are development sets now; v9 will be written later and stays sealed).
The twins below are the builder's own code, written from the class descriptions, not copies of dev-set files.

## Hypotheses (one per implemented class)

1. **Two-sided transforms** (`filtfilt`, `sosfiltfilt`, `hpfilter`, `savgol_filter`, `gaussian_filter1d`,
   `np.gradient`, `argrelextrema`/`argrelmin`/`argrelmax`/`find_peaks`, `np.convolve`/`scipy.signal.convolve` with
   `mode="same"`, a Kalman `.smooth(...)`, `seasonal_decompose`/`STL`, a rolling window whose `center` comes from a
   config default) read later bars. High when the result reaches a signal, position, feature or model; possible
   otherwise (a plot or a printed diagnostic).
2. **Wrong-line findings**: the same leak is also reported where it is written (the feature-list constant, the test
   scoring call inside a selection loop, the `max(results, key=...)` pick, the `direction=`/`right_on=` keyword line, the
   negative shift inside a helper).
3. **Test-set pickers**: `max`/`min` over candidates keyed on a metric of test-named data, a dict of test scores picked
   with `max(d, key=d.get)`, and an Optuna objective returning a test-set score are LA14 high.
4. **CV and sampling**: `TimeSeriesSplit` without `gap` when the label looks more than one bar ahead (medium);
   `fit_resample` (SMOTE and similar) before the split (high); `.sample(frac=1)` or torch `random_split` on time-ordered
   rows before a time-series split or model (medium); shuffled K-fold also recognized when the file shows forward labels;
   a grid search fitted on all rows and then scored on a slice of the same rows (medium).
5. **Survivorship** (SB01 medium): `dropna(axis=1)` (any, or `thresh=len(...)`) on a wide price or return frame; columns
   kept by the last row (`iloc[-1].notna()`); a universe taken from the last row (`iloc[-1].nlargest(n)`, latest market
   cap); winners by full-period return; current members (`removed`/`end_date` `.isna()`, a "current" constituents file);
   `groupby(id).filter(lambda g: g[date].max() >= END)`.
6. **Period aggregates broadcast to earlier bars**: `groupby(<day/week/month derived from the index>)[col].transform(f)`
   or `pd.Grouper(freq=...)` with f not in (first, count, cumulative): LA12 medium when the result reaches a signal or
   feature.
7. **Fills at the bar's own low/high/VWAP**: a fill or entry price taken from the same bar's low or high (`entry =
   bar["low"]`, `np.where(sig, low, nan)`, `from_signals(price=low)`); a limit set from bar t's close checked against bar
   t's own low; VWAP treated like the open for same-bar timing.
8. **Structural look-ahead**: backtrader `self.data.<field>[k]` with k > 0; a shift helper called with a negative lag;
   a negative shift at the end of a groupby-transform lambda that reaches a model input (high, was possible).
9. **Full-history per-asset statistics**: `groupby(<asset>)[col].rank(pct=True)` reaching a signal (LA09 medium);
   winsorization at full-sample quantiles and `fillna(frame.mean()/median())` before a later date split that feeds a model
   (LA09 medium).
10. **Full-sample estimation driving the backtest** (LA05 medium): PCA, `KMeans`, `sm.OLS`, `np.polyfit`, cointegration
    tests, `arch_model`, `ARIMA` fitted on all rows (no split visible) whose output reaches positions, weights or PnL.
11. **False alarms**: clean_16's class (names derived from bars of the same resample rule keep the bars' stamps), and
    the count-only trade cap (Y04: `len(trades)` and emptiness checks read how many fills happened, not their values).

## Corpus

Written: 47 leaky and 33 clean twins (one or more per class above, plus clean twins for the two false-alarm classes and a value-based trade cap as the leaky twin of Y04).

`tests/corpus_v06`: `leaky/` files with `# expect: RULE[:severity][@line]` (`A|B` = either rule; `@line` = a high/medium
finding within 2 lines), `clean/` files with `# expect: none`. Every number on it is in-sample.

## Acceptance criteria (fixed before measuring)

1. Clean v0.6 twins: 0 with a high or medium finding.
2. Leaky v0.6 twins: all flagged as expected.
3. Dev sets v2-v8: **no new false alarm** on any correct file (v8's 4 false alarms may only go down), and no leaky
   file caught by v0.5.0 is lost. Report recall per set before and after (in-sample).
4. Every existing test corpus: 0 high/medium on clean files, no lost catch against main.
5. Full test suite green; perf within the budgets in the tests (3 s at 200 KB, 4 s for the 232-268 KB red-team shapes,
   8 s ceiling on every run).
6. Rescan this repository and the earlier internal codebase against main; every added high/medium is reviewed and
   reported.
7. Rule of doubt: when the flow condition can't be shown, "possible", never high.

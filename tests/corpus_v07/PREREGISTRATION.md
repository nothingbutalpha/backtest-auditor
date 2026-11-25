# Lint v0.7 (recall sprint): pre-registration

Committed before any change to `lint.py` and before lint v0.6 (main) was run on these twins. The miss classes come from
`docs/RECALL_ANALYSIS_v07.md`. v2-v9 are development sets (v9's sealed run is recorded in
our internal sealed-run record); every number on them, on these twins and on rt14-rt16 is in-sample. A public
number needs a sealed v10. The twins are the builder's own code, written from the class descriptions, not copies of
dev-set files.

## Hypotheses (what is attempted)

1. **Interprocedural flow (class A).** A helper (a `def`, a function passed to `.pipe` or `.apply`, or a class whose
   constructor stores statistics) whose returned value is built from a whole-sample statistic or fit of its own
   argument (mean/std/quantile/median/min/max, a scaler, imputer or selector `fit`/`fit_transform`, `polyfit`,
   mean/covariance weights, a reversed expanding window, a two-sided filter, a per-group `apply` lambda doing the
   same) is judged at its call sites. When a call passes a full frame (not a training slice or a split-named frame,
   not inside a per-window loop, not per date) and the result reaches a model that is later split or cross-validated,
   or a signal, position or PnL, the finding is reported where the statistic is written (LA09 or LA05 medium; LA02
   high for two-sided filters as in v0.6). Future-derived series that pass through a helper's local names, a returned
   tuple, a class attribute or a dict of frames are followed the same way (LA01/LA10 at the negative shift).
2. **Survivorship (class B, SB01 medium).** `dropna(axis=1)` (any, or `thresh=len(...)`) on a wide price frame reached
   through `.loc[...]`, a `.pipe(lambda d: d.dropna(...))` or a resample before the returns; `cols[F.iloc[-1].notna()]`
   (also in a helper); a universe cut from the last row by a threshold (`last[last > x].index`); current members
   (`end`/`removed`/`delist` column `.isna()`) inside a helper whose result picks the names per date, or in a
   `query("<col>.isnull()")` string; a constituents file named as a current snapshot (`current`, `latest`, `today`)
   whose names select the columns of a backtested price frame (was "possible"); `last_valid_index` equal to the last
   date. Each needs the selected names to reach a backtest (time-series operations, positions or PnL).
3. **Selection on the full sample (class C, LA05 medium).** Features chosen on all rows (`corrwith(y)` then
   `nlargest`/`head`/`sort_values` index; `SelectKBest(...).fit(X, y)`; `mutual_info_*`/`f_regression`) and then
   used in a cross-validation, a walk-forward loop or a later date split; pairs or spreads chosen by a full-sample
   `coint`/`adfuller` test in a loop and then backtested over the same rows; min-variance or mean-variance weights
   from the full-sample covariance applied to the same returns. Not flagged when the selection runs on a training
   slice, inside a pipeline, or only prints.
4. **Wrong line (class D).** A forward column in a feature-list constant is also reported at the list (LA10, `same_as`
   the fit line); LA15 is also reported at the `right_on=`/`left_on=` keyword line of a `merge_asof` when that is a
   different line (`same_as` the call line).
5. **CV and walk-forward (class E, LA06 medium).** `cross_val_score`/`cross_validate`/`cross_val_predict` with an
   integer `cv` or an unshuffled `KFold` while the label looks more than one bar ahead; a hand-written walk-forward
   loop (`train = df.iloc[:k]`, `test = df.iloc[k:k + step]`) with no purge gap for a multi-bar label; the v0.6
   `TimeSeriesSplit`-without-gap rule extended to labels whose horizon is a module constant; a grid search fitted on
   all rows and then used on a later slice.
6. **Point patterns (class F).** `groupby(level=<asset>).rank(pct=True)` (LA09); a per-day aggregate keyed by a day
   column assigned from the index (`df["day"] = df.index.date`) (LA12); a pooled `pd.qcut`/`quantile` clip over all
   dates of a panel that is then grouped by date (LA09); `percentileofscore` against a full-history array (LA09); a
   limit set from a bar's close and checked against that bar's low in an `iterrows`/`iloc` row loop (LA07);
   statsmodels `smoothed_state` (LA02); an all-time high or low (`px.max()`) driving a signal (LA09). For LA12 and
   LA02 flow severity, a signal that selects or groups a forward return (`fwd[sig].mean()`,
   `groupby(sig)[fwd].mean()`) counts as a decision.

Not attempted: class G (interpolation stays "possible" by policy; point-in-time stamps, news timing, time zones,
corporate actions, target encoding, meta-label features, event windows, dollar-bar thresholds, intrabar stop/target
order, trailing-stop ratchets, window slices that include the target).

## Corpus

`tests/corpus_v07`: 43 leaky and 41 clean twins (at least one leaky and one correct look-alike per hypothesis).
`leaky/` files start with `# expect: RULE[|RULE][:severity]@line` (`A|B` = either rule; a high/medium finding within 2
lines of `line`); `clean/` files with `# expect: none`. The clean twins include the near-miss forms each rule must
leave alone (training-slice fits, walk-forward helpers, per-date transforms, trailing windows, live-only universes,
reports and plots, purged loops, `gap=` splits, point-in-time rosters).

## Measurement (fixed before measuring)

Per dev set v2-v9 and rt14, rt15, rt16, before (v0.6 at 57879a8) and after, each set scored by its own scorer
(`score.py`; v2 `score_env.py`; rt14-16 `score_rt14.py`): recall, recall by difficulty and by category (v9's scorer
output), false alarms split into plain and near-miss with the manifest labels (the same split as `ops/eval/fp_split.py`).
All of it labelled in-sample.

## Acceptance criteria

1. Clean v0.7 twins: 0 with a high or medium finding.
2. Leaky v0.7 twins: report how many are flagged as expected (target: all; a twin that can't be met is reported, not
   edited).
3. Dev sets v2-v9 and rt14-rt16: **no new false alarm** on any correct file, plain or near-miss (the existing ones may
   only go down), and **no lost catch**: every leaky file caught by v0.6 is still caught.
4. Every existing test corpus: 0 high/medium on clean files and no lost catch; the full auditor suite green.
5. Speed: the existing perf tests pass; every 200 KB input stays under the 8 s ceiling, including the red team's
   stress shapes and new shapes for the v0.7 rules (helper call sites, many helpers, many `dropna`/selection loops).
6. Static only: no new imports, no execution of the linted code.
7. Rule of doubt: when the flow condition can't be shown, "possible", never high or medium.
8. The package version is not changed in this sprint.

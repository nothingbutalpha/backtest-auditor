# Lint v0.4 (documented misses into rules): pre-registration

Committed before any change to `lint.py` and before the v0.3 lint (main at b5e8d1c) was run on these files.

## Hypotheses

v0.3 documents these misses and false alarms. Each can become a rule (or a principled exemption) without adding a
high or medium finding to correct code:

1. **Same-bar fill timing (LA07).** An entry or exit priced at bar t's open, selected by a signal that uses bar t's
   close, high, low or volume (`open.where(sig)`, `open.mask(~sig)`, `open[sig]`, `df.loc[sig, "open"]`,
   `np.where(sig, open, nan)`, `df.loc[sig, "entry_px"] = df["open"]`), and bar loops where the decision variable was
   set from bar t's close earlier in the iteration and the fill reads `open[t]`. Clean: next bar's open
   (`open.shift(-1)`), the same bar's close, a lagged signal, a signal known at the open (a gap from the prior close).
2. **Full-sample normalization inside `apply`/`transform`/`pipe` (LA09).** A lambda, a named helper or `zscore`
   that scales each column by its own whole-column mean/std/min/max/abs-max, on a time-series frame. Clean:
   `groupby(<date>)`, `axis=1`, rolling/expanding windows, statistics of a training slice, one date's cross-section,
   a transposed frame.
3. **Pivot keeps the forward flag (LA10/LA01).** `pivot(values=...)`, `pivot_table(values=...)` (also positional) and
   `unstack()` of a forward column stay forward, with the same lead. Clean: the pivoted forward return used only as the
   outcome (booked against an earlier position, an IC, a decile spread, an equal-weight benchmark), or lagged by its
   full horizon.
4. **`query`/`eval` strings (LA10).** Literal expression strings are parsed and their names read as columns of the
   frame (`@name` as a variable, backticks as a column), so a product or a signal written in a string is judged like
   the same code written out. Clean: strings over past-only columns, outcome products, date filters, capture
   statistics.
5. **`np.copysign(r, outcome)` (LA10).** The sign argument is a position; when it is future data and the magnitude is a
   return, it is a position times a return. Clean: the sign from a past-only signal (the forward return in the
   magnitude is the booked outcome), a label `copysign(1, fwd)`, an error metric.
6. **LA13 on our own code.** (a) An offset that resolves to a negative number (a module constant, or a parameter
   default that is never reassigned) is a past bar, not a future one. (b) A loop over a string's characters
   (`t[i + 1] == " "`, `t` visibly a `str`) is not a bar loop. (c) A later bar read only inside a missing-data check
   (`np.isnan`, `np.isfinite`, `pd.isna`, `.isna()`, `is None`) is "possible", not high: it can still skip trades
   before a delisting, so it is not exempt.

## Corpus

`tests/corpus_v04` was written by the builder from the backlog descriptions only. The sealed sets and the red-team
probe folders were not read. Every clean file is a near miss for at least one leaky twin. It is a development set:
**every number on it is in-sample.**

- `leaky/`: 31 files, each with `# expect: RULE[:severity]` (`A|B` = either rule). Prefixes: `a` fill timing (7),
  `b` normalization in apply (6), `c` pivot (6), `d` query/eval (6), `e` copysign (3), `f` LA13 (3).
- `clean/`: 36 files, `# expect: none` (no high or medium finding, any rule): `a_c` 7, `b_c` 8, `c_c` 6, `d_c` 7,
  `e_c` 4, `f_c` 4.
- Some leaky files may already be caught by v0.3 (for example `a05`, a decision on a precomputed close signal); they
  stay in the set as regression cases. The baseline is measured after this commit.

## Acceptance criteria (fixed before measuring)

1. Clean v0.4 dev files: **0** with any high or medium finding.
2. Leaky v0.4 dev files: **all** flagged with the expected rule at the expected severity.
3. Every existing clean corpus keeps 0 high or medium findings (`corpus`, `corpus_v02`, `corpus_v02_holdout`,
   `corpus_rt2`, `corpus_rt3`, `corpus_rt5`, `corpus_v03*`).
4. No lost true catch against main: every leaky file or block that main flags at high or medium is still flagged at
   high or medium (the same rule or another rule that names the same leak), across every test corpus.
5. The full test suite stays green. Documented misses that become catches (rt6 A04-A06, rt7 E01/E04, rt9 W06) move
   out of the KNOWN_MISSES lists; none is required.
6. Performance: 200 KB of code lints in under 3 s locally (the test keeps the 8 s ceiling).
7. Rescan our internal research code and an earlier internal codebase against main; report every added and removed high or
   medium finding. A removed one must be a confirmed false alarm on review, otherwise it is reported as a
   regression. Our own LA13 highs in an internal research module (`volume_peak_offsets`) and
   `nba_auditor/path_audit.py:84` go away or drop to "possible" only through the principled rules in
   hypothesis 6; if no such rule fits, they are documented instead.
8. Rule of doubt: where a new rule can't tell whether a value reaches a trade, it reports "possible", never high.

## Addendum 1 (coordinator request, before measuring these files): three false-alarm classes from a fresh sealed set

Received mid-build. The descriptions came from the coordinator; the sealed set itself was not read. These twins were
written and this addendum committed before v0.3 or v0.4 was run on them. Precision on these classes comes before any
new recall rule above.

7. **Fill-condition models (LA01/LA10).** A limit order placed on bar t's signal rests during the next bar only;
   whether it filled is decided by the next bar's low (buy) or high (sell) compared with the limit price
   (`filled = want_buy & (low.shift(-1) <= limit_px)`), and trade returns are then selected with that fill mask. The
   next-bar low/high used only for the fill decision is the fill model, not a signal input. Twins: `g_c01`-`g_c03`
   clean; `g01` (the next low in the decision to place the order), `g02` (a next-low dip used as the signal), `g03`
   (the fill flag used as a model feature) leaky.
8. **Label-only loops (LA13).** A loop that reads later bars (`close.iloc[i + 1: i + 1 + horizon]`) to build labels
   or a target that is returned and used only as the label (y of a fit, `label=`) or summarised, never as a feature,
   signal or position, is a label builder, not look-ahead. Twins: `h_c01`-`h_c03` clean; `h01` (the same
   triple-barrier output used as a feature), `h02` (a forward label traded as a position) leaky.
9. **Bar construction with default resample labels (LA12).** Ticks resampled to 1-minute or 5-minute OHLC, volume
   and VWAP with the default labels, then saved, printed or summarised, and never joined back onto finer data or
   another time series, is not a misalignment. Twins: `i_c01`-`i_c03` clean; `i01` (bars reindexed onto ticks),
   `i02` (bars joined onto a fills frame) leaky, both expected at medium.

Acceptance: the criteria above apply to these files too (0 clean with high or medium; every leaky twin flagged as
expected), plus: none of the three exemptions may drop an existing catch in any corpus.

## Addendum 2 (coordinator feedback on the v0.4 change, before measuring these files)

The coordinator scored the change on the held-out sets (the builder did not see them): one false alarm remains, class 8.
A labeler inside a function returns `labels.iloc[: len(close) - horizon]`; the caller assigns `y = ...` and uses `y`
only for summary statistics (`value_counts`, `describe`), never in a fit. Extension: a loop's forward reads are exempt
when the result reaches only labels or targets **or pure evaluation and summary sinks** (`value_counts`, `describe`,
`mean`, `print`, `plot`, `to_csv`/`to_parquet`), and never a feature, signal, position or model input. There must
still be positive evidence that the output is a label (a label-named output, call-site target or function name).
Twins: `h_c04`, `h_c05` clean; `h03` (summarised, then used as a feature), `h04` (summarised, then sets a position)
leaky. Acceptance: as above, 0 clean with high or medium, both leaky flagged, every other result unchanged.

## Addendum 3 (coordinator feedback on 98d1c91, before measuring these files)

The held-out case is still LA13 high: the caller aligns the features to the labels' dates
(`X = pd.DataFrame({...}).loc[y.index].dropna()`, then `print(y.loc[X.index].value_counts(normalize=True))`).
Index-only uses of the labels (`y.index`, `.loc[y.index]` / `reindex(y.index)` on another frame,
`X.index.intersection(y.index)`, `len(y)`, `y.shape`) are alignment, not a value read, and are neutral for the
label-only exemption. Value uses in the features (`X["lab"] = y`, `concat([X, y])`, `y.values`) still break it.
Twins: `h_c06` clean (exactly this alignment), `h05` leaky (y concatenated into X, then fitted). Acceptance as above.

Not decided here: the sealed-set result and the red-team pass (RETROS #23). Both need people and sets this build has
not seen.

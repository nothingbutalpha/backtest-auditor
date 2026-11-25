# Lint v0.3 (LA10 precision): pre-registration

Committed before the v0.2 lint was run on this corpus and before any change to `lint.py`.

## Hypothesis

The remaining high false alarms of v0.2 on correct code come from LA10 (target leakage) in three classes:

1. Forward-looking columns (`fwd_close = g.shift(-1)`, `fwd_ret`, `label`) that are dropped from the feature
   matrix before fitting, including drop lists built by a comprehension over `df.columns` filtered by prefix.
2. Forward returns used only as the evaluation or holding-period return (decile spreads, IC,
   `groupby([date, decile])[fwd_ret].mean()`).
3. Next-open execution prices (`entry_px = open.shift(-1)`, `exit_px = open.shift(-2)`) with a signal that only
   uses data up to t.

LA10 can be made precise on these classes without losing the leaks it catches today.

## Corpus

`tests/corpus_v03` was written by the builder from the task's class descriptions only (the sealed sets
heldout_v2/v3/v4 were not read). Every clean file has at least one leaky twin that differs in the leak.
It is a development set: every number on it is **in-sample**.

- `clean/`: 26 files (9 class 1, 9 class 2, 8 class 3), expected `none`.
- `leaky/`: 19 files, each with `# expect: RULE[:severity]`; `A|B` means either rule.

## Acceptance criteria (fixed before measuring)

1. Clean dev files: **0** with any high or medium finding (any rule).
2. Leaky dev files: **19/19** flagged with the expected rule at the expected severity.
3. No regression: every existing corpus test (`corpus`, `corpus_v02`, `corpus_v02_holdout`, `corpus_rt2`,
   `corpus_rt3`) and every existing LA10 expectation still passes; the full suite stays green.
4. Performance: the 200 KB input still lints in under 3 s (`test_large_input_is_fast`).
5. Rescans of our internal code: report every finding added or removed against v0.2. Any removed high or medium
   finding must be a confirmed false alarm on review; otherwise it is reported as a regression.
6. Rule of doubt: where the lint can't tell whether a forward column reaches a feature, it reports "possible", not
   high.

## Addendum (added mid-build, before measuring these files): LA05 and CO01

Requested by the coordinator after the LA10 work above had been built and measured. The files in
`tests/corpus_v03b` were written and this addendum committed before v0.2 or v0.3 was run on them.

- **LA05 hypothesis.** v0.2 raises LA05 on correct code whose training subset is a slice
  (`.loc[:cut]`, `.iloc[:k]`, `[:k]`, `.loc[start:end]`) or a boolean date mask, not a variable named `train`.
  Corpus: 9 clean (`d01`-`d09`), 7 leaky twins that fit on the full data and evaluate an overlapping later slice.
- **CO01 hypothesis.** Missing costs is a realism caveat, not a lookahead red flag. CO01 is capped at "possible"
  everywhere. Corpus: 3 clean files (`k01`-`k03`) that must have no high or medium finding; `k03` must still show
  CO01 as "possible".
- **Acceptance:** 0/12 clean files with any high or medium finding; 7/7 leaky twins with LA05 at high or medium
  (the `d08` twin at high); CO01 never above "possible" anywhere in the test corpora; criteria 3-6 above apply too.

## Addendum 2 (review feedback, before measuring these files)

`tests/corpus_v03c`: 7 clean, 7 leaky twins for three cases found after the PR opened. (1) A negative shift at the
end of a chain (`rolling/ewm/pct_change ... .shift(-k)`) flowing through features into a signal lagged by fewer
bars than the lead. (2) A negative `np.roll` used as a comparison value feeding a position. (3) A position times a
return between next-bar fill prices (the execution model, clean), against a signal that reads the fill prices
(leaky). Acceptance: 0/7 clean with any high or medium finding; 7/7 leaky with LA01 or LA10 at high; criteria
1-6 above still hold on every other corpus.

The sealed-set gate for the free web check (0 high false alarms on a fresh sealed set) is not decided here: it
needs a set this build has never seen.

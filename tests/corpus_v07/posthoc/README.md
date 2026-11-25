# Post-hoc twins (not pre-registered)

Added after the pre-registration commit (ce67afd), at the coordinator's request, from two lint gaps found while
building the draft_backtest templates (`test_known_gap_*` strict xfails):

1. vectorbt entries that are not shifted, filled with `from_signals(..., price=open)`: the signal uses bar t's close
   but fills at bar t's open.
2. A signal returned by a helper function and traded from the same bar's open.

The twins are the builder's own code, written from those two descriptions (not copies of the templates). They were
written before any lint change, but after the pre-registration, so they are post-hoc and in-sample like every other
v0.7 dev file.

## Post-hoc rule extensions (after reading the dev-set misses)

Four extensions are not covered by the pre-registered hypotheses. They were added after reading the remaining v4-v9
misses, so their dev-set catches are post-hoc (in-sample twice over) and are reported separately:

- P1: LA14 on a selection loop is also reported at the test-scoring call inside the loop (`same_as`).
- P2: LA10 on a fit is also reported where a forward-looking feature it uses is defined (`same_as`).
- P3: a shuffled K-fold is flagged (LA06 medium) when the label is forward-named and the file has no time-series hint.
- P5: a dict comprehension of test-set scores picked with `max(d, key=d.get)` is LA14 high.
- P6 (recorded after the red-team review, internal record, finding F1): the k-fold CV rule
  (`cross_val_score`/`cross_validate`/`cross_val_predict` with integer, default or unshuffled folds) also fires on a
  1-bar forward label. Pre-registered hypothesis 5 said "more than one bar ahead". It is kept deliberately (our
  decision): k-fold CV on a time series trains on folds that come after the test fold. v3 leaky_21 is caught only
  because of it, and it may cost a plain false alarm in v10 (rt17 q17 is that shape).

`phc03`-`phc06` are their correct look-alikes, written by the builder after the rules (post-hoc).

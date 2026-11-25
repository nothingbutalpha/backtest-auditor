# Lint v0.5 (sealed v7 false alarms): pre-registration

Committed before any change to `lint.py` and before v0.4 (main at a8d57fc, package 0.4.0) was run on these files.

## Background

The pre-registered one-shot run of v0.4 on sealed set v7 flagged 4 of 90 correct files at high (all near-miss: three
LA13 loops, one LA07). The coordinator described three classes from that review. The builder has not read v7, any
`heldout_*` folder, or v8 (which will be written later and stays sealed).

## Hypotheses

1. **Loop order simulation (LA13).** A bar loop that decides an order at bar i from data up to bar i's close, then
   checks bar i+1's low (high) against the order's limit, stop or take-profit level to see whether it filled, and
   prices the fill from that level (and bar i+1's open), is the loop form of the vectorized fill model. It is not
   look-ahead. Exempt only when all of these hold:
   - the condition is the comparison of bar i+1's low or high (exactly one bar ahead) with a level known before bar
     i+1's high and low happen (past data, or bar i+1's open), possibly `and`-ed with past-only terms;
   - the order was decided without future data (an enclosing condition or an `and` term on past data only, or an
     `elif` of another fill check);
   - the branch prices the fill or exit from that level (`min(limit, o[i + 1])`, `exit_px = tp`);
   - every later-bar read in the branch is a price assigned to a fill, exit or PnL name (no size, weight or position).
   When the level isn't used in the fill price, the finding is "possible", not exempt.
2. **numpy label loops ending in numpy summary or file sinks (LA13).** The label-only exemption also accepts
   `np.unique`, `np.bincount`, `np.histogram`, `np.save`/`savez`/`savetxt`, `.tofile`, nan-aware means and counts, a
   comparison with a constant inside such a summary (`np.count_nonzero(labels == 1)`), and missing-value masks
   (`valid = ~np.isnan(labels)`), which read no label values. Any value use in features, signals or products still
   breaks it.
3. **Lags inside `groupby(...).transform(lambda s: ...)` (LA07).** A positive shift at the end of the lambda's chain
   (also `.apply`, `.pipe`, and named helper functions that return `param....shift(k)`) lags the result, as does
   `x.pct_change().groupby(keys).shift(1)`.

## Corpus

`tests/corpus_v05`, written by the builder from these descriptions. Every number on it is **in-sample**.
- `leaky/`: 8 files with `# expect: RULE[:severity]` (`A|B` = either rule): `m01`-`m04` order simulation (the next
  bar's high decides the entry, a next-bar close filter, a fill at bar i's own open, sizing from the next bar's
  range), `n01`-`n02` numpy labels used as a position and as a feature, `p01`-`p02` transforms with no lag at the end.
- `clean/`: 10 files with `# expect: none` (no high or medium finding): `m_c01`-`m_c04`, `n_c01`-`n_c02`,
  `p_c01`-`p_c04`.

## Acceptance criteria (fixed before measuring)

1. Clean v0.5 twins: 0 with a high or medium finding.
2. Leaky v0.5 twins: all flagged as expected.
3. Every existing clean corpus keeps 0 high or medium findings; no high or medium catch is lost against main on any
   corpus (`tests/`, `eval/`, `examples/`, every red-team block).
4. Full test suite green; perf: 200 KB under 3 s locally, every red-team perf shape under 3 s, 8 s ceiling.
5. Rescan this repository and the earlier internal codebase against main and report every added and removed high or
   medium finding; a removed one must be a confirmed false alarm on review.
6. Rule of doubt: ambiguous order-simulation loops are "possible", not exempt.

## Addendum 1 (coordinator scoring of d0dbece; before measuring these files)

v7 is now a dev set. Two v7 false alarms remain, both loop order simulations in idioms the first build misses. The
coordinator gave the exact shapes; these twins were written from them and committed before the fix.

4. **Inside a function, configured by a dataclass, looping over precomputed signal indices.** The order is decided by
   the indices (`for i in signal_idx`, computed from past data), a bounds check `if i + 1 + cfg.hold_bars >= len(c):
   break` reads no data, and the exit price `c[i + 1 + cfg.hold_bars]` of the filled position is booked only into the
   trade's own return and log (`trades.append((bars.index[i + 1], fill, exit_px / fill - 1))`). Exit prices read
   after the fill bar are treated like fill prices when they only reach the trade's return or log.
5. **The next bar read through a row alias.** `nxt = bars.iloc[t + 1]` (also `x[t + 1]`, `.loc`) is tracked as bar
   t+1; `nxt["high"]`/`nxt["low"]`/`nxt["open"]` get the same order-simulation treatment as `high[t + 1]` etc. An
   entry price taken from `nxt["open"]` is a fill price, not a decision input.

Twins: `q_c01`, `q_c02` clean; `q01` (signal indices from future data), `q02` (the next close decides inside the
simulation), `q03` (`nxt["close"]` decides the entry), `q04` (`nxt["high"]` decides the entry) leaky. A past-only
`and` term counts as the order decision only when it reads bar data (`not in_pos` alone is position state, not a
decision). Acceptance: as above.

Optional: the documented misses H2 and G13, only if cheap and safe. Not decided here: the sealed v8 result and the
red-team pass.

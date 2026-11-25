"""Red-team round 10 (PR #15, lint v0.4, BLOCKED at 6659762): regression cases.

tests/corpus_rt10 holds the red team's repro blocks (`#### NAME` then the code). Blocks starting with N or P are
correct code; every other block is a leak. Required (regressions against v0.3): L1 (a label written as a column,
then lagged into a signal), L3 (a label list extended into positions), G2 (the next bar's low used for sizing next
to a fill mask), G7 (an entry priced at the next bar's low next to a fill mask). Also caught since this round: H1,
L2, L5, F3. Perf shapes are in test_rt10_perf_* below. Documented misses (README, Known limitations) are listed in
KNOWN_MISSES.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from nba_auditor import check_code

D = Path(__file__).parent / "corpus_rt10"
B: dict[str, str] = {}
for _fn in sorted(D.glob("*.txt")):
    B.update(dict(b.split("\n", 1) for b in re.split(r"^#### ", _fn.read_text(encoding="utf-8"), flags=re.M)[1:]))


def _hm(src: str) -> set[str]:
    return {f.rule_id for f in check_code(src).findings if f.severity in ("high", "medium")}


REQUIRED = ["L1_label_column_then_lagged_signal", "L3_label_list_extend_into_positions", "G2_fill_feeds_sizing",
            "G7_fill_named_filled_but_entry_at_low"]
# H2: a loop with no condition (label_max[i] = c[i + 1: i + 21].max() / c[i] - 1) is not LA13's pattern, and the
# value isn't tracked as forward. H5: forward values written by a helper loop, then smoothed into a signal.
# J5: forward columns merged in from a second frame. Q1: `strategy = df.query("fwd_ret > 0")` then a mean
# (a capture-like statistic, not counted).
# Round 11 (r11.txt, APPROVE at 1fb35fc on condition of G9): G13 is a numpy fill model with offset slices
# (`lo[1:] <= lim[:-1]`, sizing from `lo[1:]`), which the fill-model rules don't line up.
KNOWN_MISSES = {"H2_label_named_feature", "H5_label_loop_values_into_signal", "J5_merge_labels_frame_list",
                "Q1_query_strategy_name", "G13_fill_mask_via_numpy"}
CLEAN = sorted(k for k in B if k[0] in "NP")
LEAKY = sorted(k for k in B if k[0] not in "NP" and k not in KNOWN_MISSES)


def test_rt10_corpus_loaded():
    assert set(REQUIRED) <= set(B) and len(CLEAN) >= 22 and len(LEAKY) >= 30


@pytest.mark.parametrize("name", LEAKY)
def test_rt10_leak_is_flagged(name):
    assert _hm(B[name]), name


@pytest.mark.parametrize("name", CLEAN)
def test_rt10_correct_code_has_no_high_or_medium(name):
    assert not _hm(B[name]), (name, _hm(B[name]))


def test_rt10_known_misses_are_still_misses():
    """Listed in the README's Known limitations. Move a name out of KNOWN_MISSES (and the README) when caught."""
    now_caught = sorted(n for n in KNOWN_MISSES if _hm(B[n]))
    assert not now_caught, now_caught


# ------------------------------------------------------------------ perf shapes from the red team (local budget 3 s)
_HDR = 'import pandas as pd, numpy as np\nc = pd.read_csv("x.csv")["close"].to_numpy()\n'


def _labeler_funcs(n: int) -> str:
    return _HDR + "".join(
        f"def make_labels{k}(c, h=5):\n    out = np.zeros(len(c))\n    for i in range(len(c) - h):\n        v = c[i + h]\n"
        f"        if v > c[i]:\n            out[i] = 1\n    return out\ny{k} = make_labels{k}(c)\nprint(y{k}.mean())\n"
        for k in range(n))


def _label_loops(n: int) -> str:
    return _HDR + "".join(
        f"labels{k} = []\nfor i in range(len(c) - 5):\n    x = c[i + 5]\n    if x > c[i]:\n        labels{k}.append(1)\n"
        f"print(np.mean(labels{k}))\nlabels{k} = labels{k}\n" for k in range(n))


def _fill_models(n: int) -> str:
    return _HDR + 'df = pd.DataFrame({"close": c, "low": c, "high": c})\n' + "".join(
        f'lim{k} = df["close"] * 0.99\nf{k} = (df["close"] < df["close"].rolling({k % 20 + 2}).mean()) & '
        f'(df["low"].shift(-1) <= lim{k})\nr{k} = (df["close"].shift(-5) / lim{k} - 1)[f{k}]\n' for k in range(n))


@pytest.mark.parametrize("shape,n", [("labeler_funcs", 1100), ("label_loops", 1600), ("fill_models", 1600)])
def test_rt10_perf_shapes(shape, n):
    import os

    # Red-team worst-case shapes (232-268 KB, above the 200 KB size the 3 s budget is set for): local budget 4 s (about
    # 3 s scaled by size, plus margin), the median of three runs when close, CI 5 s, and the 8 s web check timeout on
    # every run. On this laptop main itself measures 2.2-3.4 s on these shapes depending on load.
    HARD_CEILING_S, BUDGET_S = 8.0, (5.0 if os.environ.get("CI") else 4.0)
    src = {"labeler_funcs": _labeler_funcs, "label_loops": _label_loops, "fill_models": _fill_models}[shape](n)
    def timed() -> float:
        t0 = time.perf_counter()
        assert check_code(src).parsed
        return time.perf_counter() - t0

    first = timed()
    assert first < HARD_CEILING_S, f"{shape} ({len(src) // 1000} KB): {first:.2f}s"
    # the median of three runs against the local budget: a busy machine adds noise to single runs (rt12 review)
    if first >= BUDGET_S * 0.8:
        rest = [timed(), timed()]
        assert max(rest) < HARD_CEILING_S, f"{shape}: {max(rest):.2f}s reaches the {HARD_CEILING_S:.0f}s ceiling"
        dt = sorted([first, *rest])[1]
    else:
        dt = first
    assert dt < HARD_CEILING_S, f"{shape} ({len(src) // 1000} KB): {dt:.2f}s"
    assert dt < BUDGET_S, f"{shape} ({len(src) // 1000} KB): {dt:.2f}s over {BUDGET_S}s ({'CI' if os.environ.get('CI') else 'local'})"

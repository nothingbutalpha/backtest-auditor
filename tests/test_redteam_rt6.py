"""Red-team round 6 (review of PR #9 at 0ba08c2, verdict APPROVE WITH FIXES): regression cases.

tests/corpus_rt5/r6_leaky.txt and r6_clean.txt hold the red team's repro blocks (`#### NAME` then the code).
Required: A01, A03 and C07 (regressions against v0.2). Recommended: B01-B05 (identical-expression cancel scope),
K01 and K09 (false alarms). A04-A05 (`pivot(values="fwd_ret")`) are caught since v0.4, which carries the forward
flag through pivot; A06 (the product inside a helper `backtest(w, r)`) and B05 are documented misses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

RT = Path(__file__).parent / "corpus_rt5"


def _blocks(fn: str) -> dict[str, str]:
    txt = (RT / fn).read_text(encoding="utf-8")
    return dict(b.split("\n", 1) for b in re.split(r"^#### ", txt, flags=re.M)[1:])


def _hm(src: str) -> set[str]:
    return {f.rule_id for f in check_code(src).findings if f.severity in ("high", "medium")}


L, C = _blocks("r6_leaky.txt"), _blocks("r6_clean.txt")
# A06: the position x return product happens inside a helper (`backtest(w, r)`), which the outcome rules don't
# follow. B05: the helper that reads df["fwd"] is defined above the line that creates it; the data-flow pass runs in
# source order (v0.2 misses it too). A04/A05 (pivot) are caught since v0.4.
KNOWN_MISSES = {"A06_ls_helper_backtest", "B05_param_default_shadow"}


@pytest.mark.parametrize("name", ["A04_long_short_weights_panel_mul", "A05_long_short_weights_panel_mul_method"])
def test_rt6_pivot_keeps_the_forward_flag(name):
    assert any(f.rule_id == "LA10" and f.severity == "high" for f in check_code(L[name]).findings), name


@pytest.mark.parametrize("name", sorted(set(L) - KNOWN_MISSES))
def test_rt6_leak_is_flagged(name):
    assert _hm(L[name]), name


@pytest.mark.parametrize("name", sorted(C))
def test_rt6_correct_code_has_no_high_or_medium(name):
    assert not _hm(C[name]), (name, _hm(C[name]))


def test_rt6_known_misses_are_still_misses():
    """Helper products and source-order helpers (README, Known limitations). Update both if this changes."""
    now_caught = sorted(n for n in KNOWN_MISSES if _hm(L[n]))
    assert not now_caught, now_caught

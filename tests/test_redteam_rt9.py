"""Red-team round 9 (PR #9 at 6dbc2b4, APPROVE conditional on documentation): regression cases.

tests/corpus_rt5/r9.txt: V* reorder a container's columns through expressions the lint can't read as a literal
list (order must become unknown), W* select returns with a future-data mask, X* are correct code.
W06 (`frame.query("m > 0")["b"]` stored as `strat`) was a documented miss in v0.3; since v0.4 query strings are
parsed and a strategy-named series kept only on rows chosen by future data is medium.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

B = dict(b.split("\n", 1) for b in re.split(
    r"^#### ", (Path(__file__).parent / "corpus_rt5" / "r9.txt").read_text(encoding="utf-8"), flags=re.M)[1:])


def _sev(src: str) -> set[str]:
    return {f.severity for f in check_code(src).findings if f.severity in ("high", "medium")}


CLEAN = ["X01_clean_capture_inline", "X02_clean_lagged_regime_mask"]
KNOWN_MISSES: set[str] = set()  # v0.4: W06 is caught (medium)
LEAKY = sorted(set(B) - set(CLEAN) - KNOWN_MISSES)


@pytest.mark.parametrize("name", LEAKY)
def test_rt9_leak_is_flagged(name):
    assert _sev(B[name]), name


@pytest.mark.parametrize("name", CLEAN)
def test_rt9_correct_code_has_no_high_or_medium(name):
    assert not _sev(B[name]), (name, _sev(B[name]))


def test_rt9_neutral_mask_is_medium_not_high():
    """ok = mkt > 0; bench[ok]: a neutral name can be an up/down-capture filter, so medium, not high."""
    assert _sev(B["W01_neutral_mask"]) == {"medium"}


def test_rt9_query_strategy_is_medium():
    assert _sev(B["W06_query_on_frame"]) == {"medium"}


def test_rt9_known_misses_are_still_misses():
    now_caught = sorted(n for n in KNOWN_MISSES if _sev(B[n]))
    assert not now_caught, now_caught

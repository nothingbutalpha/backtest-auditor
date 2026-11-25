"""Red-team round 7 (review of PR #9 at c8bfe78): regression cases.

tests/corpus_rt5/r7.txt and r7b.txt hold the red team's repro blocks. E* are outcome-driven positions written in
other forms, F*/H* are container column renames, G* are correct code. np.copysign (E01) and frame.eval("t * b")
(E04) were documented misses in v0.3 and are caught since v0.4.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

RT = Path(__file__).parent / "corpus_rt5"
B: dict[str, str] = {}
for _fn in ("r7.txt", "r7b.txt"):
    _txt = (RT / _fn).read_text(encoding="utf-8")
    B.update(dict(b.split("\n", 1) for b in re.split(r"^#### ", _txt, flags=re.M)[1:]))


def _hm(src: str) -> set[str]:
    return {f.rule_id for f in check_code(src).findings if f.severity in ("high", "medium")}


CLEAN = sorted(n for n in B if n.startswith("G"))
KNOWN_MISSES: set[str] = set()  # v0.4: E01 (np.copysign) and E04 (frame.eval) are now caught
LEAKY = sorted(set(B) - set(CLEAN) - KNOWN_MISSES)


@pytest.mark.parametrize("name", LEAKY)
def test_rt7_leak_is_flagged(name):
    assert _hm(B[name]), name


@pytest.mark.parametrize("name", CLEAN)
def test_rt7_correct_code_has_no_high_or_medium(name):
    assert not _hm(B[name]), (name, _hm(B[name]))


def test_rt7_known_misses_are_still_misses():
    """Kept for the pattern: anything listed here must stay a documented miss in the README."""
    now_caught = sorted(n for n in KNOWN_MISSES if _hm(B[n]))
    assert not now_caught, now_caught

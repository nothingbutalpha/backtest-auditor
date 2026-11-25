"""Red-team round 8 (review of PR #9 at 9e6fc0a): regression cases.

tests/corpus_rt5/r8.txt: S* reorder a container's columns before a positional selection (or lag it partly),
T* use an outcome-derived long/short mask on a return, U* are correct code (U01: a container lagged by its full lead).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

B = dict(b.split("\n", 1) for b in re.split(
    r"^#### ", (Path(__file__).parent / "corpus_rt5" / "r8.txt").read_text(encoding="utf-8"), flags=re.M)[1:])


def _hm(src: str) -> set[str]:
    return {f.rule_id for f in check_code(src).findings if f.severity in ("high", "medium")}


CLEAN = sorted(n for n in B if n.startswith("U"))
LEAKY = sorted(set(B) - set(CLEAN))


@pytest.mark.parametrize("name", LEAKY)
def test_rt8_leak_is_flagged(name):
    assert _hm(B[name]), name


@pytest.mark.parametrize("name", CLEAN)
def test_rt8_correct_code_has_no_high_or_medium(name):
    assert not _hm(B[name]), (name, _hm(B[name]))

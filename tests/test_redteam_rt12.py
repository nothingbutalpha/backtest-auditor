"""Red-team round 12 (PR #18, lint v0.5, BLOCKED at e07293a): regression cases.

tests/corpus_v05/rt12_leaky.txt and rt12_clean.txt hold the red team's repro blocks (`#### NAME` then the code).
Required: R01/R02 (signal indices built from future data), R03/R04 (fill or exit prices flowing into a decision),
R06 (a next-bar low that decides a new entry), R11 (a fill priced at the next bar's low itself), and S02 (labels saved
and loaded back as a signal). Should fix: K01 (buy-stop entry), K02 (trailing stop in a while loop), K07 (a lag helper
returning `(a - b).shift(1)`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

D = Path(__file__).parent / "corpus_v05"
B: dict[str, str] = {}
for _fn in ("rt12_leaky.txt", "rt12_clean.txt"):
    B.update(dict(b.split("\n", 1) for b in re.split(r"^#### ", (D / _fn).read_text(encoding="utf-8"), flags=re.M)[1:]))


def _hm(src: str) -> set[str]:
    return {f.rule_id for f in check_code(src).findings if f.severity in ("high", "medium")}


# R10 sets a limit order's level from bar i+1's open (`limit = o[i + 1] * 0.99`) and books the fill at that limit.
# A limit placed at the next bar's open is executable, so it is a deliberate exemption (README), not a missed leak.
DELIBERATE = {"R10_limit_from_next_open"}
REQUIRED = ["R01_signal_idx_from_future_indirect", "R02_signal_idx_from_future_where", "R03_cooldown_from_trade_return",
            "R04_reentry_on_exit_px", "R06_nxt_low_new_entry_decision", "R11_fill_at_low_not_limit",
            "S02_numpy_label_save_load_signal"]
LEAKY = sorted(k for k in B if k[0] in "RSTU" and k not in DELIBERATE)
CLEAN = sorted(k for k in B if k[0] == "K")


def test_rt12_corpus_loaded():
    assert set(REQUIRED) <= set(LEAKY) and len(CLEAN) >= 9


@pytest.mark.parametrize("name", LEAKY)
def test_rt12_leak_is_flagged(name):
    assert _hm(B[name]), name


@pytest.mark.parametrize("name", CLEAN)
def test_rt12_correct_code_has_no_high_or_medium(name):
    assert not _hm(B[name]), (name, _hm(B[name]))

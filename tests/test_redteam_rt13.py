"""Red-team round 13 (PR #18, lint v0.5, BLOCK at bc95ea6, narrow): regression cases.

tests/corpus_v05/rt13_leaky.txt holds the red team's repro blocks; every block is a leak. Required: X02 (signal
indices through `self.<attr>`), X05/X07/X08 (a container holding exempt fill or exit values read in a decision).
R10 (a limit set from the next bar's open) stays exempt only on the far side of the open and booked at the limit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

D = Path(__file__).parent / "corpus_v05"
B = dict(b.split("\n", 1) for b in re.split(r"^#### ", (D / "rt13_leaky.txt").read_text(encoding="utf-8"), flags=re.M)[1:])


def _f(src: str, rule: str | None = None):
    return [(f.rule_id, f.severity, f.line) for f in check_code(src).findings if rule is None or f.rule_id == rule]


def _hm(src: str) -> set[str]:
    return {r for r, s, _l in _f(src) if s in ("high", "medium")}


@pytest.mark.parametrize("name", sorted(B))
def test_rt13_leak_is_flagged(name):
    assert _hm(B[name]), name


_R10 = ("import numpy as np\nimport pandas as pd\nd = pd.read_csv('spy.csv')\n"
        "o, h, lo, c = (d[k].to_numpy() for k in ('open', 'high', 'low', 'close'))\nout = []\n"
        "for i in range(20, len(c) - 6):\n    if c[i] < c[i - 1]:\n")


@pytest.mark.parametrize("body,exempt", [
    # a buy limit below the next open, booked at the limit: executable (round 13: R10 stays exempt)
    ("        limit = o[i + 1] * 0.99\n        if lo[i + 1] <= limit:\n            out.append(c[i + 6] / limit - 1)\n", True),
    # a sell limit above the next open, booked at the limit
    ("        limit = o[i + 1] * 1.01\n        if h[i + 1] >= limit:\n            out.append(limit / c[i + 6] - 1)\n", True),
    # a buy limit ABOVE the next open: it would fill at the open, so the condition knows the bar's low
    ("        limit = o[i + 1] * 1.01\n        if lo[i + 1] <= limit:\n            out.append(c[i + 6] / limit - 1)\n", False),
    # booked at min(limit, open) from a level set at the open: not the limit itself
    ("        limit = o[i + 1] * 0.99\n        if lo[i + 1] <= limit:\n"
     "            out.append(c[i + 6] / min(limit, o[i + 1]) - 1)\n", False),
])
def test_rt13_limit_from_next_open(body, exempt):
    got = {s for _r, s, _l in _f(_R10 + body, "LA13")}
    assert (not got & {"high", "medium"}) == exempt, got


_SIM = ("import numpy as np\nimport pandas as pd\nd = pd.read_csv('spy.csv')\n"
        "o, lo, c = (d[k].to_numpy() for k in ('open', 'low', 'close'))\n"
        "sma = d['close'].rolling(20).mean().to_numpy()\ntrades = []\nfor i in range(20, len(c) - 7):\n"
        "    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] <= limit:\n"
        "            fill = min(limit, o[i + 1])\n            trades.append(c[i + 6] / fill - 1)\n")


@pytest.mark.parametrize("tail,clean", [
    ("print(np.mean(trades), len(trades), sum(trades))\n", True),
    ("res = pd.Series(trades)\nres.to_csv('t.csv')\nprint(res.describe())\n", True),
    ("good = [t for t in trades if t > -0.02]\nprint(np.mean(good))\n", False),
    ("if np.mean(trades) > 0:\n    print('deploy')\n", False),
    # emptiness guards on a report (v7 clean_76): not decisions
    ("print(len(trades), 'trades, mean', np.mean(trades) if trades else float('nan'))\n", True),
    ("if trades:\n    print('best', max(trades))\nif len(trades) == 0:\n    print('no trades')\n", True),
    # v0.6 (Y04): how many fills happened is known in real time; a count or emptiness check is not a value read
    ("if not trades:\n    trades = [0.0]\nprint(np.mean(trades))\n", True),
    # a guard that reads the values still is
    ("if trades[-1] < 0:\n    trades = [0.0]\nprint(np.mean(trades))\n", False),
])
def test_rt13_trade_container_uses(tail, clean):
    got = {s for _r, s, _l in _f(_SIM + tail, "LA13")}
    assert (not got & {"high", "medium"}) == clean, got

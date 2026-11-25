"""check_code v0.5: the three sealed-v7 false-alarm classes (loop order simulation, numpy label sinks, lags inside
groupby transforms).

tests/corpus_v05 is the builder's development set (in-sample); acceptance criteria were committed before the lint
changed, in tests/corpus_v05/PREREGISTRATION.md. Each file starts with `# expect: RULE[:severity]` or `# expect: none`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

V05 = Path(__file__).parent / "corpus_v05"
LEAKY = sorted((V05 / "leaky").glob("*.py"))
CLEAN = sorted((V05 / "clean").glob("*.py"))


def _expect(path: Path) -> list[tuple[set[str], str]]:
    line = re.search(r"# expect: (.*)", path.read_text(encoding="utf-8")).group(1).strip()
    if line == "none":
        return []
    out = []
    for item in line.split(","):
        rules, _, sev = item.strip().partition(":")
        out.append((set(rules.split("|")), sev))
    return out


def _f(src: str, rule: str | None = None):
    return [(f.rule_id, f.severity, f.line) for f in check_code(src).findings if rule is None or f.rule_id == rule]


def _hm(src: str):
    return [(r, s, ln) for r, s, ln in _f(src) if s in ("high", "medium")]


@pytest.mark.parametrize("path", LEAKY, ids=lambda p: p.stem)
def test_v05_leak_is_flagged(path):
    got = {(r, s) for r, s, _l in _hm(path.read_text(encoding="utf-8"))}
    for rules, sev in _expect(path):
        ok = {(r, s) for r, s in got if r in rules}
        assert ok, f"{path.name}: expected {rules} at high/medium, got {sorted(got)}"
        if sev:
            assert sev in {s for _r, s in ok}, f"{path.name}: expected {rules}:{sev}, got {sorted(ok)}"


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_v05_clean_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {hm}"


def test_v05_corpus_size():
    assert len(LEAKY) >= 8 and len(CLEAN) >= 10


# ------------------------------------------------------------------ 1. loop order simulation (LA13)
_LOOP = ("import numpy as np\nimport pandas as pd\nd = pd.read_csv('spy.csv')\n"
         "o, h, lo, c = (d[k].to_numpy() for k in ('open', 'high', 'low', 'close'))\n"
         "sma = d['close'].rolling(20).mean().to_numpy()\nout = []\nfor i in range(20, len(c) - 6):\n")


@pytest.mark.parametrize("body,want", [
    # placed from past data, filled if the next low reaches the limit, priced from the limit: exempt
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] <= limit:\n"
     "            fill = min(limit, o[i + 1])\n            out.append(c[i + 6] / fill - 1)\n", set()),
    # same, but the fill is the next open whatever the limit: possible, not exempt and not high
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] <= limit:\n"
     "            fill = o[i + 1]\n            out.append(c[i + 6] / fill - 1)\n", {"possible"}),
    # an unconditional resting limit every bar, priced from the limit: executable (rt12 K01, stop entries alike)
    ("    limit = c[i] * 0.99\n    if lo[i + 1] <= limit:\n        fill = min(limit, o[i + 1])\n"
     "        out.append(c[i + 6] / fill - 1)\n", set()),
    # the same order with no decision, filled at the next open whatever the limit: high (no exemption)
    ("    limit = c[i] * 0.99\n    if lo[i + 1] <= limit:\n        fill = o[i + 1]\n"
     "        out.append(c[i + 6] / fill - 1)\n", {"high"}),
    # the low staying above the level books a trade: that's an entry decision, not a fill (rt12 R06)
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] > limit:\n"
     "            out.append(c[i + 6] / o[i + 1] - 1)\n", {"high"}),
    # skipping an order that didn't fill is fine
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] > limit:\n            continue\n"
     "        out.append(c[i + 6] / min(limit, o[i + 1]) - 1)\n", set()),
    # priced at the next bar's low itself (rt12 R11)
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] <= limit:\n"
     "            out.append(c[i + 6] / lo[i + 1] - 1)\n", {"high"}),
    # the fill price feeds a later decision (rt12 R03/R04)
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] <= limit:\n"
     "            fill = min(limit, o[i + 1])\n            if c[i + 6] > fill:\n"
     "                out.append(c[i + 6] / fill - 1)\n", {"high"}),
    # the level itself is taken from the next bar's close: not known when the order rests
    ("    if c[i] < sma[i]:\n        if lo[i + 1] <= c[i + 1] * 0.99:\n            fill = min(c[i] * 0.99, o[i + 1])\n"
     "            out.append(c[i + 6] / fill - 1)\n", {"high"}),
    # two bars ahead is not a resting order for the next bar
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 2] <= limit:\n"
     "            fill = min(limit, o[i + 2])\n            out.append(c[i + 6] / fill - 1)\n", {"high"}),
    # the fill bar's volume sizes the trade
    ("    if c[i] < sma[i]:\n        limit = c[i] * 0.99\n        if lo[i + 1] <= limit:\n"
     "            fill = min(limit, o[i + 1])\n            weight = d['volume'].to_numpy()[i + 1] / 1e6\n"
     "            out.append(weight * (c[i + 6] / fill - 1))\n", {"high"}),
])
def test_loop_order_simulation(body, want):
    got = {s for _r, s, _l in _f(_LOOP + body, "LA13")}
    assert got & {"high", "medium", "possible"} == want, got


_ALIAS = ("import pandas as pd\nbars = pd.read_csv('h.csv')\nclose = bars['close']\nres = []\nin_pos = False\n"
          "for t in range(20, len(bars) - 1):\n    nxt = bars.iloc[t + 1]\n"
          "    if not in_pos and close.iloc[t] < close.iloc[t - 12:t].min():\n")


@pytest.mark.parametrize("body,want", [
    # target from the next open (known before the bar's high), exit priced from it: exempt (addendum 1)
    ("        entry = nxt['open']\n        target = entry * 1.02\n        in_pos = True\n    elif in_pos:\n"
     "        if nxt['high'] >= target:\n            res.append(max(target, nxt['open']) / entry - 1)\n"
     "            in_pos = False\n", set()),
    # the level comes from the next bar's close: not known while the order rests
    ("        entry = nxt['open']\n        target = nxt['close'] * 1.02\n        in_pos = True\n    elif in_pos:\n"
     "        if nxt['high'] >= target:\n            res.append(max(target, nxt['open']) / entry - 1)\n"
     "            in_pos = False\n", {"high"}),
    # a fill check on the next low, but the entry is the next open whatever the level: possible (rule of doubt)
    ("        if nxt['low'] < close.iloc[t] * 0.98:\n            res.append(close.iloc[t + 1] / nxt['open'] - 1)\n",
     {"possible"}),
    # the next bar's low decides the trade with no order placed from past data
    ("        pass\n    if nxt['low'] < close.iloc[t] * 0.98:\n        res.append(close.iloc[t + 1] / nxt['open'] - 1)\n",
     {"high"}),
])
def test_next_bar_row_alias(body, want):
    got = {s for _r, s, _l in _f(_ALIAS + body, "LA13")}
    assert got & {"high", "medium", "possible"} == want, got


def test_signal_index_loop_in_function_with_config():
    src = ("import numpy as np\nimport pandas as pd\nHOLD = 5\n\ndef sim(bars, idx):\n"
           "    o, lo, c = bars['open'].to_numpy(), bars['low'].to_numpy(), bars['close'].to_numpy()\n    out = []\n"
           "    for i in idx:\n        if i + 1 + HOLD >= len(c):\n            break\n        limit = c[i] * 0.995\n"
           "        if lo[i + 1] <= limit:\n            fill = min(limit, o[i + 1])\n            exit_px = c[i + 1 + HOLD]\n"
           "            out.append((bars.index[i + 1], exit_px / fill - 1))\n    return out\n\n"
           "bars = pd.read_parquet('b.parquet')\nidx = np.flatnonzero((bars['close'] < bars['close'].rolling(20).mean()).to_numpy())\n"
           "print(sim(bars, idx))\n")
    assert not _hm(src)
    # the indices come from future data (rt12 R01/R02): no exemption, the fill check is high
    leak = src.replace("(bars['close'] < bars['close'].rolling(20).mean())", "(bars['close'].shift(-5) > bars['close'])")
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(leak)}
    leak2 = src.replace("idx = np.flatnonzero(", "c0 = bars['close'].to_numpy()\nidx = np.flatnonzero(c0[5:] / c0[:-5] > 1.01) if True else np.flatnonzero(")
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(leak2)}


# ------------------------------------------------------------------ 2. numpy label sinks (LA13)
_NP = ("import numpy as np\np = np.load('p.npy')\nlabels = np.zeros(len(p))\nfor i in range(len(p) - 5):\n"
       "    labels[i] = 1 if p[i + 5] > p[i] else 0\n")


@pytest.mark.parametrize("tail,leak", [
    ("print(np.count_nonzero(labels > 0), np.bincount(labels.astype(int)))\n", False),
    ("ok = ~np.isnan(labels)\nnp.savetxt('l.txt', labels[ok])\n", False),
    ("ret = np.r_[np.diff(p) / p[:-1], 0.0]\nprint((labels * ret).sum())\n", True),
    ("signal = labels > 0\n", True),
])
def test_numpy_label_sinks(tail, leak):
    got = {s for _r, s, _l in _f(_NP + tail, "LA13")}
    assert ("high" in got) == leak, got


# ------------------------------------------------------------------ 3. lags inside groupby transforms (LA07)
_PANEL = ("import pandas as pd\npanel = pd.read_parquet('p.parquet')\n"
          "panel['ret'] = panel.groupby('symbol')['close'].pct_change()\n")


@pytest.mark.parametrize("feat,lagged", [
    ("panel.groupby('symbol')['close'].transform(lambda s: s.pct_change(5).shift(1))", True),
    ("panel.groupby('symbol')['close'].transform(lambda s: s.pct_change(5).shift(1).rolling(3).mean())", True),
    ("panel.groupby('symbol')['close'].apply(lambda s: s.pct_change(5).shift(1))", True),
    ("panel.groupby('symbol')['close'].transform(lambda s: s.pct_change(5))", False),
    ("panel.groupby('symbol')['close'].transform(lambda s: s.pct_change(5) - s.shift(1))", False),
])
def test_lag_inside_transform(feat, lagged):
    src = _PANEL + f"panel['mom'] = {feat}\npanel['w'] = (panel['mom'] > 0).astype(float)\n" \
                   "pnl = (panel['w'] * panel['ret']).groupby(panel['date']).sum()\n"
    got = {s for _r, s, _l in _f(src, "LA07")}
    assert ("high" in got) != lagged, got

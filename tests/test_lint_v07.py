"""check_code v0.7 (recall sprint). tests/corpus_v07 is the builder's development set (in-sample); the acceptance
criteria and twins were committed before the lint changed (tests/corpus_v07/PREREGISTRATION.md). The `posthoc` folder
holds twins added after the pre-registration (draft_backtest gaps from PR #40), labelled post-hoc.
Each file starts with `# expect: RULE[|RULE][:severity]@line` or `# expect: none`; `@line` means a high/medium
finding within 2 lines of that line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

V07 = Path(__file__).parent / "corpus_v07"
LEAKY = sorted((V07 / "leaky").glob("*.py"))
CLEAN = sorted((V07 / "clean").glob("*.py"))
POSTHOC_LEAKY = sorted((V07 / "posthoc" / "leaky").glob("*.py"))
POSTHOC_CLEAN = sorted((V07 / "posthoc" / "clean").glob("*.py"))


def _expect(path: Path) -> list[tuple[set[str], str, int | None]]:
    line = re.search(r"# expect: (.*)", path.read_text(encoding="utf-8")).group(1).strip()
    if line == "none":
        return []
    out = []
    for item in line.split(","):
        item, _, at = item.strip().partition("@")
        rules, _, sev = item.partition(":")
        out.append((set(rules.split("|")), sev, int(at) if at else None))
    return out


def _hm(src: str):
    return [(f.rule_id, f.severity, f.line) for f in check_code(src).findings if f.severity in ("high", "medium")]


def _check_leaky(path: Path) -> None:
    got = _hm(path.read_text(encoding="utf-8"))
    for rules, sev, at in _expect(path):
        ok = [g for g in got if g[0] in rules and (not sev or g[1] == sev) and (at is None or abs(g[2] - at) <= 2)]
        assert ok, f"{path.name}: expected {rules}:{sev or 'high/medium'} near line {at}, got {sorted(got)}"


@pytest.mark.parametrize("path", POSTHOC_LEAKY, ids=lambda p: p.stem)
def test_v07_posthoc_leak_is_flagged(path):
    _check_leaky(path)


@pytest.mark.parametrize("path", POSTHOC_CLEAN, ids=lambda p: p.stem)
def test_v07_posthoc_clean_has_no_high_or_medium(path):
    assert not _hm(path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("path", LEAKY, ids=lambda p: p.stem)
def test_v07_leak_is_flagged_on_its_line(path):
    _check_leaky(path)


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_v07_clean_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {hm}"


def test_v07_corpus_size():
    assert len(LEAKY) >= 43 and len(CLEAN) >= 41


# -- speed: 200 KB shapes that exercise the v0.7 flow index (helper call sites, shared method names, survivorship,
#    pair loops, CV calls, dict entries, reversed windows). Each must stay under the web check's 8 s ceiling. --------
_HDR = ("import numpy as np\nimport pandas as pd\nfrom statsmodels.tsa.stattools import coint\n"
        "from sklearn.model_selection import cross_val_score\npx = pd.read_parquet('close.parquet')\nret = px.pct_change()\n")
_SHAPES = {
    "helper stats": "def z{k}(s):\n    mu, sd = s.mean(), s.std()\n    return (s - mu) / sd\nx{k} = z{k}(px)\n"
                    "sig{k} = np.sign(x{k}).shift(1)\npnl{k} = sig{k} * ret\n",
    "classes same method": "class C{k}:\n    def __init__(self, d):\n        self.m = d.mean()\n    def tr(self, b):\n"
                           "        return b - self.m\nc{k} = C{k}(px)\nX{k} = c{k}.tr(px)\nprint(X{k})\n",
    "one helper many sites": "x{k} = h(px)\nprint(x{k}.tail())\n",
    "dropna survivors": "p{k} = px.loc['2001':].dropna(axis=1)\nm{k} = p{k}.resample('ME').last()\nr{k} = m{k}.pct_change()\n",
    "coint loops": "pairs{k} = []\nfor a, b in [('A', 'B')]:\n    _, p{k}, _ = coint(px[a], px[b])\n    if p{k} < 0.05:\n"
                   "        pairs{k}.append((a, b))\n",
    "cv calls": "y{k} = (df['ret_fwd_{k}'] > 0).astype(int)\ns{k} = cross_val_score(m, X, y{k}, cv=5)\n",
    "dict entries": "b{k} = {{}}\nb{k}['lead{k}'] = px.shift(-1)\nX{k} = pd.concat(b{k}, axis=1)\nm.fit(X{k}, y)\n",
    "reverse windows": "r{k} = px.iloc[::-1]\nmu{k} = r{k}.expanding().mean().iloc[::-1]\nsig{k} = (px > mu{k}).shift(1)\n",
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_v07_stress_shapes_stay_under_the_ceiling(shape):
    import time

    from test_lint_v02 import HARD_CEILING_S

    head = _HDR + ("def h(s):\n    return (s - s.mean()) / s.std()\n" if shape == "one helper many sites" else "")
    src, k = [head], 0
    while sum(map(len, src)) < 200_000:
        src.append(_SHAPES[shape].format(k=k))
        k += 1
    t = time.perf_counter()
    check_code("".join(src))
    dt = time.perf_counter() - t
    assert dt < HARD_CEILING_S, f"{shape}: {dt:.2f}s reaches the {HARD_CEILING_S:.0f}s ceiling"

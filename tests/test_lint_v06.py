"""check_code v0.6 (recall sprint): two-sided transforms, findings on the line where the leak is written, test-set
pickers, CV and sampling leaks, survivorship patterns, period aggregates, fills at the bar's own low/high/VWAP,
structural look-ahead, per-asset full-history statistics, full-sample estimates that set positions; plus two false
alarms (bars derived from the same resample rule; a count-only trade cap).

tests/corpus_v06 is the builder's development set (in-sample); the acceptance criteria were committed before the lint
changed, in tests/corpus_v06/PREREGISTRATION.md. Each file starts with `# expect: RULE[|RULE][:severity][@line]` or
`# expect: none`; `@line` means a high/medium finding within 2 lines of that line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code
from nba_auditor.lint import score_findings

V06 = Path(__file__).parent / "corpus_v06"
LEAKY = sorted((V06 / "leaky").glob("*.py"))
CLEAN = sorted((V06 / "clean").glob("*.py"))


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


def _f(src: str, rule: str | None = None):
    return [(f.rule_id, f.severity, f.line) for f in check_code(src).findings if rule is None or f.rule_id == rule]


def _hm(src: str):
    return [(r, s, ln) for r, s, ln in _f(src) if s in ("high", "medium")]


@pytest.mark.parametrize("path", LEAKY, ids=lambda p: p.stem)
def test_v06_leak_is_flagged_on_its_line(path):
    got = _hm(path.read_text(encoding="utf-8"))
    for rules, sev, at in _expect(path):
        ok = [(r, s, ln) for r, s, ln in got if r in rules and (not sev or s == sev) and (at is None or abs(ln - at) <= 2)]
        assert ok, f"{path.name}: expected {rules}:{sev or 'high/medium'} near line {at}, got {sorted(got)}"


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_v06_clean_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {hm}"


def test_v06_corpus_size():
    assert len(LEAKY) >= 40 and len(CLEAN) >= 30


# Post-hoc probes (written after the fixes; not part of the pre-registered twins): correct look-alikes.
POSTHOC = dict(b.split("\n", 1) for b in re.split(
    r"^#### ", (V06 / "posthoc_clean.txt").read_text(encoding="utf-8"), flags=re.M)[1:])


@pytest.mark.parametrize("name", sorted(POSTHOC))
def test_v06_posthoc_lookalikes_are_clean(name):
    hm = _hm(POSTHOC[name])
    assert not hm, f"{name}: false alarms {hm}"


# -- repeats of a finding where the leak is written are shown, but scored once ------------------------------------
_SWEEP = (
    "import numpy as np\nimport pandas as pd\npx = pd.read_parquet('p.parquet')['close']\nret = px.pct_change()\n"
    "scores = {}\nfor lb in range(20, 260, 20):\n    pos = np.sign(px.pct_change(lb)).shift(1)\n    pnl = pos * ret\n"
    "    scores[lb] = pnl.mean() / pnl.std()\nbest = max(scores, key=scores.get)\nprint(best)\n"
)


def test_v06_repeat_is_not_scored_twice():
    a = check_code(_SWEEP)
    la11 = [f for f in a.findings if f.rule_id == "LA11"]
    assert sorted(f.line for f in la11) == [6, 10]
    rep = next(f for f in la11 if f.line == 10)
    assert rep.same_as == 6 and rep.to_dict()["same_as"] == 6
    assert "same_as" not in next(f for f in la11 if f.line == 6).to_dict()
    assert a.counts["medium"] == 1
    assert next(b for b in a.score_breakdown if b["rule_id"] == "LA11")["findings"] == 1
    assert score_findings([f for f in a.findings if not f.same_as])[0] == a.score
    assert "same issue as line 6" in a.to_markdown()


def test_v06_helper_negative_shift_reported_inside_the_helper():
    src = ("import pandas as pd\n\n\ndef ahead(s, k):\n    return s.shift(-k)\n\n\n"
           "df = pd.read_parquet('x.parquet')\nsignal = (ahead(df['close'], 1) > df['close']).astype(int)\n"
           "print((signal.shift(1) * df['close'].pct_change()).sum())\n")
    lines = {(r, ln) for r, s, ln in _hm(src)}
    assert ("LA01", 9) in lines and ("LA01", 5) in lines


# -- two-sided transforms: high only when the value reaches a decision ----------------------------------------------
_PX = "import numpy as np\nimport pandas as pd\nfrom scipy.signal import filtfilt, butter\npx = pd.read_csv('p.csv')['close']\nb, a = butter(2, 0.1)\n"


@pytest.mark.parametrize("tail,sev", [
    ("sm = filtfilt(b, a, px.to_numpy())\nposition = np.sign(np.diff(sm, prepend=sm[0]))\n", "high"),
    ("sm = filtfilt(b, a, px.to_numpy())\nprint(sm[-5:])\n", "possible"),
    ("target = filtfilt(b, a, px.to_numpy())\n", None),  # a smoothed label: future data belongs there
    ("sm = filtfilt(b, a, px.to_numpy())\ntarget = sm\n", "possible"),  # only reaches a label: not a decision
])
def test_v06_two_sided_severity_follows_the_flow(tail, sev):
    got = {s for r, s, _l in _f(_PX + tail, "LA02")}
    assert got == ({sev} if sev else set()), got


def test_v06_rolling_center_false_from_config_is_clean():
    src = ("import pandas as pd\nclass C:\n    center = False\n\n\ndef f(s, cfg=C()):\n"
           "    return s.rolling(5, center=cfg.center).mean()\n\n\nsignal = f(pd.read_csv('x.csv')['close'])\n")
    assert not _f(src, "LA02")


# -- false alarms fixed --------------------------------------------------------------------------------------------
_Y = ("import numpy as np\nimport pandas as pd\nd = pd.read_csv('q.csv')\no, lo, c = (d[k].to_numpy() for k in ('open', 'low', 'close'))\n"
      "sma = d['close'].rolling(30).mean().to_numpy()\ntrades = []\nfor i in range(30, len(c) - 6):\n{guard}"
      "    if c[i] < sma[i]:\n        limit = c[i] * 0.995\n        if lo[i + 1] <= limit:\n"
      "            fill = min(limit, o[i + 1])\n            trades.append(c[i + 5] / fill - 1)\nprint(np.mean(trades))\n")


@pytest.mark.parametrize("guard,clean", [
    ("    if len(trades) >= 50:\n        break\n", True),           # Y04: how many fills so far
    ("    if trades and len(trades) % 10 == 0:\n        pass\n", True),
    ("    if trades and trades[-1] < -0.05:\n        break\n", False),  # the last trade's return: a value read
    ("    if sum(trades) < -0.2:\n        break\n", False),
])
def test_v06_trade_count_cap(guard, clean):
    hm = _hm(_Y.format(guard=guard))
    assert (not hm) == clean, hm


def test_v06_same_rule_bars_derived_names_join_cleanly():
    src = ("import pandas as pd\nq = pd.read_parquet('q.parquet')\na = q['x'].resample('1s').mean()\n"
           "m = q['mid'].resample('1s').last()\nfeat = pd.DataFrame({'a5': a.rolling(5).mean()})\n"
           "nxt = m.pct_change().shift(-1).rename('y')\nprint(feat.join(nxt).corr())\n")
    assert not [x for x in _hm(src) if x[0] == "LA12"]
    # a derived name joined onto a *different* index still realigns the bars
    src2 = src.replace("print(feat.join(nxt).corr())", "ticks = pd.read_parquet('t.parquet')\nprint(ticks.join(feat).corr())")
    assert [x for x in _hm(src2) if x[0] == "LA12"]


# -- red-team round 14 (internal review record): fixes F1-F5 -----------------------------------------------
# corpus_rt14_v06 was written by the red team; every number on it after these fixes is in-sample.
RT14 = Path(__file__).parent / "corpus_rt14_v06"
RT14_KNOWN = {"q10_next_bar_limit_named": {"LA10"}, "q18_smote_after_split": {"LA05"}}  # pre-v0.6, out of scope


@pytest.mark.parametrize("path", sorted((RT14 / "clean").glob("*.py")), ids=lambda p: p.stem)
def test_rt14_correct_files(path):
    hm = [x for x in _hm(path.read_text(encoding="utf-8")) if x[0] not in RT14_KNOWN.get(path.stem, set())]
    assert not hm, f"{path.name}: false alarms {hm}"


@pytest.mark.parametrize("path", sorted((RT14 / "leaky").glob("*.py")), ids=lambda p: p.stem)
def test_rt14_leaks(path):
    # k09 (a universe from last_valid_index) was a documented miss until v0.7, which catches it
    assert _hm(path.read_text(encoding="utf-8")), path.name


_BT = ("import backtrader as bt\nclass S(bt.Strategy):\n    params = (('periods', (10, 30)),)\n"
       "    def __init__(self):\n        self.sma = bt.ind.SMA(self.data.close, period={per})\n"
       "        self.levels = [0.98, 1.02]\n    def next(self):\n        if {cond}:\n            self.buy()\n")


@pytest.mark.parametrize("per,cond,leak", [
    ("self.p.periods[1]", "self.data.close[0] > self.sma[0]", False),        # F1: params tuple
    ("self.params.periods[1]", "self.data.close[0] > self.sma[0] * self.levels[1]", False),
    ("10", "self.data.close[1] > self.sma[0]", True),                        # next bar of a data line
    ("10", "self.sma[1] > self.sma[0]", True),                               # next value of an indicator
    ("10", "self.datas[0].close[1] > 0", True),
])
def test_rt14_backtrader_lines_only(per, cond, leak):
    got = [x for x in _hm(_BT.format(per=per, cond=cond)) if x[0] == "LA13"]
    assert bool(got) == leak, got


def test_rt14_shifted_low_in_a_variable_is_next_bar():
    src = ("import pandas as pd\nb = pd.read_csv('b.csv')\nlimit = b['close'] * 0.99\nnxt_low = b['low'].shift(-1)\n"
           "filled = nxt_low <= limit\nfill_ret = (b['close'].shift(-3) / limit - 1).where(filled)\nprint(fill_ret.mean())\n")
    assert not [x for x in _hm(src) if x[0] == "LA07"]


@pytest.mark.parametrize("src,sev", [
    ("import numpy as np, pandas as pd\ns = pd.read_csv('x.csv')['close']\n"
     "slope = pd.Series(np.gradient(s.to_numpy())).shift(1)\nsignal = (slope > 0).astype(int)\n", "possible"),  # lagged by the look-ahead
    ("import numpy as np, pandas as pd\ns = pd.read_csv('x.csv')['close']\n"
     "slope = pd.Series(np.gradient(s.to_numpy()))\npos = np.sign(slope).shift(1)\npnl = pos * s.pct_change()\n", "high"),  # one lag is used up by the position
    ("import numpy as np, pandas as pd\ns = pd.read_csv('x.csv')['close']\n"
     "slope = pd.Series(np.gradient(s.to_numpy()))\nsignal = (slope > 0).astype(int)\n", "high"),
    ("import numpy as np, pandas as pd\nfrom scipy.signal import argrelextrema\ns = pd.read_csv('x.csv')['close']\n"
     "k = 7\nm = pd.Series(0, index=s.index)\nm.iloc[argrelextrema(s.to_numpy(), np.less, order=k)[0]] = 1\n"
     "signal = m.shift(k)\n", "possible"),
    ("import numpy as np, pandas as pd\nfrom scipy.signal import argrelextrema\ns = pd.read_csv('x.csv')['close']\n"
     "k = 7\nm = pd.Series(0, index=s.index)\nm.iloc[argrelextrema(s.to_numpy(), np.less, order=k)[0]] = 1\n"
     "signal = m.shift(3)\n", "high"),                                     # lagged less than the look-ahead
    ("import pandas as pd\nfrom scipy.signal import savgol_filter\ns = pd.read_csv('x.csv')['close']\n"
     "sm = s.rolling(21).apply(lambda w: savgol_filter(w, 21, 3)[-1], raw=True)\nsignal = (s > sm).astype(int)\n", None),
])
def test_rt14_causal_uses_of_two_sided_tools(src, sev):
    got = {s for r, s, _l in _f(src, "LA02")}
    assert got == ({sev} if sev else set()), got


def test_rt14_reported_period_total_is_not_a_signal():
    src = ("import pandas as pd\nm = pd.read_parquet('m.parquet')\nm['pos'] = (m['close'] > m['close'].rolling(20).mean()).shift(1)\n"
           "m['pnl'] = m['pos'] * m['close'].pct_change()\nm['day_pnl'] = m.groupby(m.index.date)['pnl'].transform('sum')\n"
           "print(m['day_pnl'].describe())\n")
    assert not [x for x in _hm(src) if x[0] == "LA12"]


def test_rt14_stress_shapes_are_fast():
    """F5: two 200 KB shapes that hit quadratic paths in b090f0e (12.5 s and 7.0 s)."""
    from test_lint_v02 import _assert_fast, _timed

    hdr = "import numpy as np, pandas as pd\nfrom scipy.signal import butter, filtfilt\nb, a = butter(2, 0.1)\nprices = pd.read_parquet('p.parquet')\n"
    unit = "def sm{k}(x):\n    return filtfilt(b, a, x)\n\n\nsignal{k} = np.sign(np.diff(sm{k}(prices['c{k}'].to_numpy())))\n"
    src, k = [hdr], 0
    while sum(map(len, src)) < 200_000:
        src.append(unit.format(k=k))
        k += 1
    _assert_fast("helper filtfilt", _timed("".join(src)))
    unit2 = "px{k} = prices.dropna(axis=1)\nr{k} = px{k}.pct_change()\n"
    src, k = [hdr], 0
    while sum(map(len, src)) < 200_000:
        src.append(unit2.format(k=k))
        k += 1
    _assert_fast("dropna columns", _timed("".join(src)))
    unit3 = "b['limit{k}'] = b['close'] * 0.99\nb['hit{k}'] = b['low'] <= b['limit{k}']\n"  # found while fixing F5
    src, k = [hdr], 0
    while sum(map(len, src)) < 200_000:
        src.append(unit3.format(k=k))
        k += 1
    _assert_fast("limit compares", _timed("".join(src)))


# -- red-team round 15 (B1, B2): corpus_rt15_v06, in-sample after these fixes -----------------------------------------
RT15 = Path(__file__).parent / "corpus_rt15_v06"
RT15_KNOWN = {"n04_limit_next_bar_column": {"LA10"}}  # LA10 already on main, out of scope


@pytest.mark.parametrize("path", sorted((RT15 / "clean").glob("*.py")), ids=lambda p: p.stem)
def test_rt15_correct_files(path):
    hm = [x for x in _hm(path.read_text(encoding="utf-8")) if x[0] not in RT15_KNOWN.get(path.stem, set())]
    assert not hm, f"{path.name}: false alarms {hm}"


@pytest.mark.parametrize("path", sorted((RT15 / "leaky").glob("*.py")), ids=lambda p: p.stem)
def test_rt15_leaks_on_their_line(path):
    src = path.read_text(encoding="utf-8")
    line = int(re.search(r"# leak: (\d+)", src).group(1))
    assert any(abs(ln - line) <= 2 for _r, _s, ln in _hm(src)), (path.name, _hm(src))


def test_rt15_aggregate_of_prices_feeding_exits_is_a_decision():
    src = ("import numpy as np, pandas as pd\nm = pd.read_parquet('m.parquet')\n"
           "m['cap'] = m.groupby(m.index.date)['low'].transform('min') * 1.001\n"
           "exit_px = np.where(m['low'] <= m['cap'], m['cap'], m['close'])\nm['pnl'] = exit_px / m['open'] - 1\n")
    assert [x for x in _hm(src) if x[0] == "LA12"]

"""check_code v0.4: documented misses turned into rules, and three sealed-set false-alarm classes.

tests/corpus_v04 is the builder's development set (in-sample). The acceptance criteria were committed before any
lint change, in tests/corpus_v04/PREREGISTRATION.md (items 1-6, then addendum 1 for items 7-9).

Each file starts with `# expect: RULE[:severity]` (`A|B` = either rule) or `# expect: none`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

V04 = Path(__file__).parent / "corpus_v04"
LEAKY = sorted((V04 / "leaky").glob("*.py"))
CLEAN = sorted((V04 / "clean").glob("*.py"))
REPO = Path(__file__).resolve().parents[1]


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
def test_v04_leak_is_flagged(path):
    got = {(r, s) for r, s, _l in _hm(path.read_text(encoding="utf-8"))}
    for rules, sev in _expect(path):
        ok = {(r, s) for r, s in got if r in rules}
        assert ok, f"{path.name}: expected {rules} at high/medium, got {sorted(got)}"
        if sev:
            assert sev in {s for _r, s in ok}, f"{path.name}: expected {rules}:{sev}, got {sorted(ok)}"


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_v04_clean_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {hm}"


def test_v04_corpus_size():
    assert len(LEAKY) >= 40 and len(CLEAN) >= 47


# ------------------------------------------------------------------ 1. same-bar fill timing (LA07)
_RSI = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\ndf['sma'] = df['close'].rolling(20).mean()\n"
        "entries = df['close'] > df['sma']\n")


@pytest.mark.parametrize("line,sev", [
    ("entry_px = df['open'].where(entries)\n", "high"),
    ("entry_px = df['open'][entries]\n", "high"),
    ("entry_px = df.loc[entries, 'open']\n", "high"),
    ("opens_on_signal = df['open'].where(entries)\n", "possible"),  # not a fill name: possible
    ("entry_px = df['open'].shift(-1).where(entries)\n", None),  # the next bar's open
    ("entry_px = df['close'].where(entries)\n", None),  # the same bar's close
    ("entry_px = df['open'].where(entries.shift(1, fill_value=False))\n", None),  # a lagged signal
])
def test_open_fill_selected_by_a_close_signal(line, sev):
    got = {s for _r, s, _l in _f(_RSI + line, "LA07")}
    assert (sev in got) if sev else not got & {"high", "medium"}, got


def test_open_fill_with_signal_known_at_the_open_is_clean():
    src = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\ngap = df['open'] / df['close'].shift(1) - 1\n"
           "entry_px = df['open'].where(gap < -0.02)\n")
    assert not _hm(src)


# ------------------------------------------------------------------ 2. normalization inside apply (LA09)
_TS = ("import pandas as pd\npx = pd.read_csv('p.csv', index_col=0)['close']\n"
       "feats = pd.DataFrame({'mom': px.pct_change(20), 'vol': px.pct_change().rolling(20).std()})\n")


@pytest.mark.parametrize("call,sev", [
    ("feats.apply(lambda c: (c - c.mean()) / c.std())", "medium"),
    ("feats.transform(lambda c: (c - c.min()) / (c.max() - c.min()))", "medium"),
    ("feats.apply(lambda c: (c - c.mean()) / c.std(), axis=1)", None),
    ("feats.rolling(20).apply(lambda w: (w - w.mean()).sum())", None),
    ("feats.T.apply(lambda c: (c - c.mean()) / c.std()).T", None),
])
def test_apply_normalization(call, sev):
    src = _TS + f"features = {call}\nsignal = (features['mom'] > 0).shift(1)\n"
    got = {s for _r, s, _l in _f(src, "LA09")}
    assert (sev in got) if sev else not got & {"high", "medium"}, got


def test_apply_normalization_without_time_series_evidence_is_possible():
    src = ("import pandas as pd\nsnap = pd.read_csv('fundamentals.csv')\n"
           "z = snap[['ey', 'bp']].apply(lambda c: (c - c.mean()) / c.std())\nscore = z.sum(axis=1)\n")
    assert not {s for _r, s, _l in _f(src, "LA09")} & {"high", "medium"}


# ------------------------------------------------------------------ 3. pivot / pivot_table / unstack (LA10)
_PANEL = ("import pandas as pd\ndf = pd.read_parquet('p.parquet')\n"
          "df['fwd_ret'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1\n")


def test_pivot_carries_the_forward_flag():
    src = _PANEL + "w = df.pivot(index='date', columns='ticker', values='fwd_ret')\nsignal = (w > 0).astype(int)\n"
    assert ("LA10", "high") in {(r, s) for r, s, _l in _f(src)}


def test_pivot_table_by_bucket_is_an_outcome_table():
    src = _PANEL + ("df['q'] = df.groupby('date')['mom'].transform(lambda s: pd.qcut(s, 5, labels=False))\n"
                    "tbl = df.pivot_table(index='date', columns='q', values='fwd_ret', aggfunc='mean')\n"
                    "q_spread = tbl[4] - tbl[0]\nprint(q_spread.mean())\n")
    assert not _hm(src)


def test_pivot_lagged_by_its_full_horizon_is_clean():
    src = _PANEL + ("w = df.pivot(index='date', columns='ticker', values='fwd_ret')\nlast = w.shift(1)\n"
                    "signal = (last > 0).astype(int)\n")
    assert not _hm(src)


# ------------------------------------------------------------------ 4. query / eval strings (LA10)
_ONE = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\ndf['fwd_ret'] = df['close'].pct_change().shift(-1)\n"
        "df['mom'] = df['close'].pct_change(20).shift(1)\n")


@pytest.mark.parametrize("line,leak", [
    ("df['signal'] = df.eval('fwd_ret > 0')\n", True),
    ("df.eval('signal = fwd_ret > 0', inplace=True)\n", True),
    ("df = df.eval('signal = fwd_ret > 0')\n", True),
    ("df['signal'] = df.eval('mom > 0')\n", False),
    ("expr = 'fwd_ret > 0'\ndf['signal'] = df.eval(expr)\n", False),  # not a literal: not parsed (a miss)
    ("df['signal'] = df.eval('fwd_ret >')\n", False),  # not valid: ignored
])
def test_eval_strings(line, leak):
    got = {(r, s) for r, s, _l in _hm(_ONE + line)}
    assert bool(got) == leak, got


def test_query_rows_by_past_data_then_book_forward_return_is_clean():
    src = _ONE + "days = df.query('mom > 0.05').index\ndf['position'] = df.index.isin(days).astype(int)\n" \
                 "df['pnl'] = df['position'] * df['fwd_ret']\n"
    assert not _hm(src)


# ------------------------------------------------------------------ 5. np.copysign (LA10)
@pytest.mark.parametrize("line,leak", [
    ("strat = np.copysign(df['fwd_ret'], df['fwd_ret'].rolling(3).mean())\n", True),
    ("strat = np.copysign(df['fwd_ret'], df['mom'])\n", False),
    ("y = np.copysign(1.0, df['fwd_ret'])\n", False),
    ("signed_err = np.copysign(np.abs(df['mom'] - df['fwd_ret']), df['fwd_ret'])\n", False),
])
def test_copysign(line, leak):
    got = _hm("import numpy as np\n" + _ONE + line)
    assert bool(got) == leak, got


# ------------------------------------------------------------------ 6. LA13 (offsets, strings, missing-data checks)
def test_la13_module_constant_and_parameter_default_offsets():
    neg = "import numpy as np\nK = -2\nc = np.load('c.npy')\nfor i in range(5, len(c)):\n    if c[i + K] > c[i]:\n        pass\n"
    assert not _f(neg, "LA13")
    pos = "import numpy as np\nK = 2\nc = np.load('c.npy')\nfor i in range(len(c) - 2):\n    if c[i + K] > c[i]:\n        pass\n"
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(pos)}


def test_la13_missing_data_check_on_a_later_bar_is_possible():
    src = ("import numpy as np\npx = np.load('p.npy')\nfor i in range(len(px) - 1):\n"
           "    if np.isnan(px[i + 1]):\n        continue\n")
    assert {s for _r, s, _l in _f(src, "LA13")} == {"possible"}


def test_la13_string_scan_is_not_a_bar_loop():
    src = ("def words(text: str):\n    out = []\n    for i in range(len(text) - 1):\n"
           "        if text[i + 1] == ' ':\n            out.append(i)\n    return out\n")
    assert not _f(src, "LA13")


@pytest.mark.parametrize("path,func", [
    ("nba_auditor/path_audit.py", "first_sentence"),
])
def test_la13_own_code_false_alarms_are_gone(path, func):
    p = REPO / path
    if not p.exists():
        pytest.skip("repository file not present")
    src = p.read_text(encoding="utf-8")
    import ast

    fd = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == func)
    hm = [(r, s, ln) for r, s, ln in _hm(src) if r == "LA13" and fd.lineno <= ln <= fd.end_lineno]
    assert not hm, hm


# ------------------------------------------------------------------ 7. fill-condition models
_FILL = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\nwant = df['close'] < df['close'].rolling(20).mean()\n"
         "limit = df['close'] * 0.99\nfilled = want & (df['low'].shift(-1) <= limit)\n")


def test_fill_mask_selecting_trade_returns_is_the_fill_model():
    assert not _hm(_FILL + "r = (df['close'].shift(-5) / limit - 1)[filled]\nprint(r.mean(), filled.mean())\n")


@pytest.mark.parametrize("use", [
    "df['signal'] = filled.astype(int)\n",  # the fill flag as a signal
    "X = df[['close']].assign(f=filled)\n",  # as a feature
    "position = filled.shift(1)\n",
])
def test_fill_mask_used_as_an_input_is_still_flagged(use):
    assert _hm(_FILL + use)


def test_next_bar_low_without_an_order_decision_is_not_a_fill_model():
    src = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\nfilled = df['low'].shift(-1) <= df['close'] * 0.99\n"
           "r = (df['close'].shift(-5) / df['close'] - 1)[filled]\n")
    assert _hm(src)


def test_two_bar_ahead_low_is_not_a_fill_model():
    src = _FILL.replace("shift(-1)", "shift(-2)") + "r = (df['close'].shift(-5) / limit - 1)[filled]\n"
    assert _hm(src)


_SF = ("import numpy as np\nimport pandas as pd\ndf = pd.read_csv('spy.csv')\n"
       "want = df['close'] < df['close'].rolling(20).mean()\nlimit = df['close'] * 0.99\n"
       "best = df[['low', 'high']].shift(-1)\n")


@pytest.mark.parametrize("tail", [
    "filled = want & (df['low'].shift(-1) <= limit)\nr = (best['high'] / limit - 1)[filled]\n",  # rt11 G9
    "filled = want & (df['low'].shift(-1) <= limit)\nr = (best.loc[:, 'high'] / limit - 1)[filled]\n",
    "filled = want & (df['low'].shift(-1) <= limit)\nr = (best.high / limit - 1)[filled]\n",
    "filled = want & (best['low'] <= limit)\nsize = np.clip((limit - best['low']) / limit * 50, 0, 1)\n"
    "r = (size * (df['close'].shift(-5) / limit - 1))[filled]\n",  # sizing from the shifted frame
])
def test_shifted_frame_column_is_that_column_shifted(tail):
    """rt11: a column of a frame shifted by -k is read exactly like the column shifted by -k."""
    assert _hm(_SF + tail)


def test_fill_mask_from_a_shifted_frame_is_still_the_fill_model():
    src = _SF + "filled = want & (best['low'] <= limit)\nr = (df['close'].shift(-5) / limit - 1)[filled]\nprint(r.mean())\n"
    assert not _hm(src)


# ------------------------------------------------------------------ 8. label-only loops (LA13)
_LAB = ("import numpy as np\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
        "def make_labels(c, h=5):\n    out = np.zeros(len(c))\n    for i in range(len(c) - h):\n"
        "        if c[i + 1: i + h].max() > c[i] * 1.02:\n            out[i] = 1\n    return out\n"
        "df = pd.read_csv('spy.csv')\ndf['mom'] = df['close'].pct_change(20)\n")


def test_label_loop_used_only_as_y_is_clean():
    src = _LAB + "y = make_labels(df['close'].to_numpy())\nLogisticRegression().fit(df[['mom']].iloc[:500], y[:500])\n"
    assert not _f(src, "LA13")


@pytest.mark.parametrize("use", [
    "df['lab'] = make_labels(df['close'].to_numpy())\nLogisticRegression().fit(df[['mom', 'lab']], df['mom'] > 0)\n",
    "y = make_labels(df['close'].to_numpy())\nsignal = y\n",
    "labels = make_labels(df['close'].to_numpy())\npnl = labels * df['close'].pct_change().shift(-1)\n",
])
def test_label_loop_output_used_otherwise_is_flagged(use):
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(_LAB + use)}


@pytest.mark.parametrize("use", [
    "y = make_labels(df['close'].to_numpy())\nprint(pd.Series(y).value_counts(normalize=True))\n",
    "y = pd.Series(make_labels(df['close'].to_numpy()))\nbalance = y.value_counts()\nbalance.plot()\ny.to_csv('y.csv')\n",
])
def test_label_loop_used_only_for_summaries_is_clean(use):
    """Addendum 2: labels that reach only summary or file sinks (value_counts, describe, plot, to_csv)."""
    assert not _f(_LAB + use, "LA13")


@pytest.mark.parametrize("use", [
    "y = make_labels(df['close'].to_numpy())\nedge = (y * df['close'].pct_change().shift(-1)).mean()\n",  # a product
    "y = make_labels(df['close'].to_numpy())\nprint(y.mean())\ndf['f'] = y\nLogisticRegression().fit(df[['mom', 'f']], y)\n",
])
def test_label_loop_summary_does_not_hide_other_uses(use):
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(_LAB + use)}


_ALIGN = _LAB + "y = pd.Series(make_labels(df['close'].to_numpy()), index=df.index)\n"


@pytest.mark.parametrize("use", [
    "X = df[['mom']].loc[y.index].dropna()\nprint(y.loc[X.index].value_counts(normalize=True))\n",
    "features = df[['mom']].reindex(y.index)\nprint(len(y), y.shape, features.shape)\n",
    "X = df[['mom']].loc[X0.index.intersection(y.index)] if False else df[['mom']]\nprint(y.mean())\n",
])
def test_label_index_alignment_is_neutral(use):
    """Addendum 3: aligning features to the labels' index reads no label values."""
    assert not _f(_ALIGN + use, "LA13")


@pytest.mark.parametrize("use", [
    "X = df[['mom']].copy()\nX['lab'] = y\nprint(X.describe())\n",
    "features = pd.concat([df[['mom']], y.rename('lab')], axis=1)\nprint(features.mean())\n",
    "features = df[['mom']].assign(lab=y.values)\nprint(features.mean())\n",
])
def test_label_values_in_features_still_break_the_exemption(use):
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(_ALIGN + use)}


def test_loop_without_label_evidence_summarised_stays_high():
    src = _LAB.replace("make_labels", "scan") + "res = scan(df['close'].to_numpy())\nprint(res.mean())\n"
    assert ("LA13", "high") in {(r, s) for r, s, _l in _f(src)}


# ------------------------------------------------------------------ 9. bar construction (LA12)
_BARS = ("import pandas as pd\nticks = pd.read_parquet('t.parquet')\nbars = ticks['price'].resample('1min').ohlc()\n"
         "bars['volume'] = ticks['size'].resample('1min').sum()\n")


def test_bars_built_and_saved_are_not_misaligned():
    assert not _hm(_BARS + "bars.to_parquet('b.parquet')\n")


def test_bars_joined_back_onto_ticks_are_medium():
    src = _BARS + "ticks['bar_close'] = bars['close'].reindex(ticks.index, method='ffill')\n"
    assert ("LA12", "medium") in {(r, s) for r, s, _l in _f(src)}


def test_resampled_column_into_the_tick_frame_is_still_medium():
    src = ("import pandas as pd\nticks = pd.read_parquet('t.parquet')\n"
           "ticks['vol_1m'] = ticks['size'].resample('1min').sum()\n")
    assert ("LA12", "medium") in {(r, s) for r, s, _l in _f(src)}

"""check_code v0.3: LA10 (target leakage) precision.

tests/corpus_v03 is the builder's development set for three LA10 false-alarm classes (forward columns dropped
before fitting, forward returns used only for evaluation, next-open execution prices) and their leaky twins.
It is in-sample. The acceptance criteria are in tests/corpus_v03/PREREGISTRATION.md.

Each file starts with `# expect: RULE[:severity]` (`A|B` = either rule) or `# expect: none`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nba_auditor import check_code

HERE = Path(__file__).parent
V03 = HERE / "corpus_v03"
V03B = HERE / "corpus_v03b"  # LA05 slice-based training subsets and CO01 severity (pre-registration addendum)
V03C = HERE / "corpus_v03c"  # round 2 (PR #9 feedback): negative-shift chains, negative np.roll, fill-price PnL
LEAKY = sorted((V03 / "leaky").glob("*.py")) + sorted((V03B / "leaky").glob("*.py")) + sorted((V03C / "leaky").glob("*.py"))
CLEAN = sorted((V03 / "clean").glob("*.py")) + sorted((V03B / "clean").glob("*.py")) + sorted((V03C / "clean").glob("*.py"))


def _expect(path: Path) -> list[tuple[set[str], str]]:
    line = re.search(r"# expect: (.*)", path.read_text(encoding="utf-8")).group(1).strip()
    if line == "none":
        return []
    out = []
    for item in line.split(","):
        rules, _, sev = item.strip().partition(":")
        out.append((set(rules.split("|")), sev))
    return out


def _hm(src: str):
    return [f for f in check_code(src).findings if f.severity in ("high", "medium")]


@pytest.mark.parametrize("path", LEAKY, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_v03_leak_is_flagged(path):
    got = {(f.rule_id, f.severity) for f in _hm(path.read_text(encoding="utf-8"))}
    for rules, sev in _expect(path):
        ok = {(r, s) for r, s in got if r in rules}
        assert ok, f"{path.name}: expected {rules} at high/medium, got {sorted(got)}"
        if sev:
            assert sev in {s for _r, s in ok}, f"{path.name}: expected {rules}:{sev}, got {sorted(ok)}"


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_v03_clean_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {[(f.rule_id, f.severity, f.line, f.message) for f in hm]}"


def test_v03_corpus_size():
    assert len(list((V03 / "leaky").glob("*.py"))) >= 19
    assert len(list((V03 / "clean").glob("*.py"))) >= 26
    assert len(list((V03B / "leaky").glob("*.py"))) >= 7
    assert len(list((V03B / "clean").glob("*.py"))) >= 12


# ------------------------------------------------------------------ targeted LA10 / LA05 behaviour (README claims)
_H = ("import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\n"
      "df = pd.read_parquet('p.parquet')\n"
      "df['fwd_ret'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1\n"
      "df['mom'] = df.groupby('ticker')['close'].pct_change(20)\n")


def _sev(src: str, rule: str = "LA10") -> dict[int, str]:
    return {f.line: f.severity for f in check_code(src).findings if f.rule_id == rule}


def test_neutral_name_is_possible_unless_it_reaches_a_model():
    label = "df['up'] = (df['fwd_ret'] > 0).astype(int)\n"
    assert _sev(_H + label) == {6: "possible"}
    fitted = _sev(_H + label + "LogisticRegression().fit(df[['mom', 'up']], df['up'])\n")
    assert fitted[6] == "medium" and fitted[7] == "high"


@pytest.mark.parametrize("use", ["df['signal'] = df['date'].map(mkt)\n", "regime = mkt > 0\n", "df['position'] = mkt\n"])
def test_outcome_aggregate_becomes_an_input_when_put_back_or_compared(use):
    src = _H + "mkt = df.groupby('date')['fwd_ret'].mean()\n" + use
    assert _sev(src).get(7) == "high"


def test_outcome_aggregate_alone_is_not_reported():
    src = _H + "mkt = df.groupby('date')['fwd_ret'].mean()\nspread = mkt - mkt.mean()\n"
    assert not {s for s in _sev(src).values()} & {"high", "medium"}


def test_unresolved_drop_list_is_possible_not_high():
    src = _H + "import json\nDROP = json.load(open('c.json'))\nX = df.drop(columns=DROP)\n"
    assert _sev(src) == {8: "possible"}


def test_drop_list_resolution():
    base = _H + "df['label'] = (df['fwd_ret'] > 0).astype(int)\n"
    ok = [
        "X = df.drop(columns=[c for c in df.columns if c.startswith('fwd_')] + ['label'])\n",
        "X = df.drop(columns=[c for c in df.columns if c.startswith(('fwd', 'lab'))])\n",
        "import re\nX = df.drop(columns=[c for c in df.columns if re.match(r'^(fwd_|label)', c)])\n",
        "X = df.loc[:, ~df.columns.str.startswith('fwd_') & (df.columns != 'label')]\n",
        "X = df.loc[:, ~df.columns.isin(['fwd_ret', 'label'])]\n",
        "X = df[[c for c in df.columns if not c.endswith('_ret') and c != 'label']]\n",
    ]
    for line in ok:
        assert not {s for s in _sev(base + line).values()} & {"high", "medium"}, line
    leaks = [
        "X = df.drop(columns=[c for c in df.columns if c.startswith('fwd_')])\n",  # keeps label
        "X = df.loc[:, ~df.columns.str.startswith('fwd_')]\n",  # keeps label
        "X = df.drop(columns=['label'])\n",  # keeps fwd_ret
    ]
    for line in leaks:
        assert "high" in _sev(base + line).values(), line


def test_next_open_fill_prices_are_not_inputs():
    src = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\n"
           "df['signal'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)\n"
           "entry_px = df['open'].shift(-1)\nexit_px = df['open'].shift(-2)\n"
           "df['pnl'] = df['signal'] * (exit_px / entry_px - 1) - 0.0005\n")
    assert not {s for s in _sev(src).values()} & {"high", "medium"}
    gap = src + "df['signal2'] = (entry_px > df['close']).astype(int)\n"
    assert "high" in _sev(gap).values()


@pytest.mark.parametrize("fit", ["X.loc[:'2018']", "X.iloc[:500]", "X.loc['2010':'2018']", "X[X.index < '2019']",
                                 "X.loc[:'2018'].to_numpy()", "scaler.transform(X.loc[:'2018'])"])
def test_la05_slice_is_a_training_subset(fit):
    src = ("import pandas as pd\nfrom sklearn.linear_model import Ridge\nfrom sklearn.preprocessing import StandardScaler\n"
           "X = pd.read_parquet('X.parquet')\ny = X.pop('y')\nscaler = StandardScaler().fit(X.loc[:'2018'])\n"
           f"m = Ridge().fit({fit}, y.loc[:'2018'])\nX_test = X.loc['2019':]\nprint(m.score(X_test, y.loc['2019':]))\n")
    assert not _sev(src, "LA05"), fit


def test_la05_full_fit_still_flagged():
    src = ("import pandas as pd\nfrom sklearn.linear_model import Ridge\n"
           "X = pd.read_parquet('X.parquet')\ny = X.pop('y')\nm = Ridge().fit(X, y)\n"
           "X_test = X.loc['2019':]\nprint(m.score(X_test, y.loc['2019':]))\n")
    assert _sev(src, "LA05") == {5: "high"}


# ------------------------------------------------------------------ round 2 (PR #9 feedback)
_P = "import numpy as np\nimport pandas as pd\ns = pd.read_csv('px.csv', index_col=0)['close']\nr = s.pct_change()\n"


@pytest.mark.parametrize("lead,lag,leaks", [(21, 1, True), (5, 1, True), (5, 5, False), (1, 1, False), (3, 4, False)])
def test_a_lag_only_undoes_a_lead_of_at_most_its_size(lead, lag, leaks):
    # neutral names (`a`, `b`): a feature-named column built with a negative shift is high by name, as in v0.2
    src = _P + (f"a = r.rolling(21).std().shift(-{lead})\nb = 0.2 - a\n"
                f"sig = (b > 0).astype(int).shift({lag})\npnl = sig * r - 0.0005\n")
    hi = {s for s in _sev(src).values()} | {s for s in _sev(src, "LA01").values()}
    assert ("high" in hi) == leaks, (lead, lag, _sev(src), _sev(src, "LA01"))


def test_unknown_negative_shift_is_never_undone():
    src = _P + "h = int(input())\nvol = r.rolling(21).std().shift(-h)\nsig = (vol > 0.01).astype(int).shift(5)\n"
    assert "high" in _sev(src).values()


@pytest.mark.parametrize("roll,leaks", [("np.roll(c, -1)", True), ("np.roll(c, shift=-2)", True), ("np.roll(c, 1)", False)])
def test_np_roll_used_as_previous_value(roll, leaks):
    src = ("import numpy as np\nc = np.load('c.npy')\nret = np.r_[0.0, c[1:] / c[:-1] - 1]\n"
           f"prev = {roll}\nprev[0] = np.nan\nup = c > prev\npos = np.zeros_like(c)\npos[1:] = np.where(up[:-1], 1, -1)\n"
           "pnl = pos * ret - 0.0005\n")
    found = set(_sev(src).values()) | set(_sev(src, "LA01").values())
    assert ("high" in found) == leaks, found


def test_position_times_fill_price_return_is_the_execution_model():
    base = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\n"
            "sig = (df['close'] > df['close'].rolling(50).mean()).astype(int)\n"
            "entry_px = df['open'].shift(-1)\nexit_px = df['open'].shift(-2)\ncosts = 0.0005 * sig.diff().abs()\n")
    assert not _sev(base + "trade = sig * (exit_px / entry_px - 1) - costs\n")
    leak = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\n"
            "entry_px = df['open'].shift(-1)\nexit_px = df['open'].shift(-2)\n"
            "sig = (entry_px > df['close']).astype(int)\ntrade = sig * (exit_px / entry_px - 1) - 0.0005\n")
    found = _sev(leak)
    assert found.get(5) == "high" and found.get(6) == "high", found


def test_label_passed_to_a_training_dataset_does_not_taint_the_model():
    src = ("import lightgbm as lgb\nimport pandas as pd\ndf = pd.read_parquet('p.parquet')\n"
           "df['fwd'] = df.groupby('ticker')['close'].shift(-20) / df['close'] - 1\n"
           "train = lgb.Dataset(df[['mom', 'value']], label=df['fwd'])\nmodel = lgb.train({}, train)\n"
           "new = pd.read_parquet('today.parquet')\nnew['score'] = model.predict(new[['mom', 'value']])\n")
    assert not _sev(src)
    leak = src.replace("df[['mom', 'value']], label", "df[['mom', 'fwd']], label")
    assert "high" in _sev(leak).values()


def test_compounding_a_fill_price_return_is_not_a_position():
    """From our internal research code (next_day_returns): (1 + r) * (1 + delisting) - 1 with r between next opens."""
    src = ("import numpy as np\nimport pandas as pd\npx = pd.read_parquet('open.parquet')\n"
           "r = px.shift(-2) / px.shift(-1) - 1.0\ndelisting_return = pd.Series(0.0, index=px.columns)\n"
           "hit = np.zeros(r.shape, dtype=bool)\nr = r.where(~hit, (1 + r) * (1 + delisting_return.to_numpy()[None, :]) - 1)\n")
    assert not {s for s in _sev(src).values()} & {"high", "medium"}


def test_co01_is_never_above_possible_and_stays_visible():
    every = sorted(HERE.glob("corpus*/*/*.py"))
    for p in every:
        sev = {f.severity for f in check_code(p.read_text(encoding="utf-8")).findings if f.rule_id == "CO01"}
        assert not sev & {"high", "medium"}, (p.name, sev)
    k03 = (V03B / "clean" / "k03_sma_total_return.py").read_text(encoding="utf-8")
    assert {f.severity for f in check_code(k03).findings if f.rule_id == "CO01"} == {"possible"}

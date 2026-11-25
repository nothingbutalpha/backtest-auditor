"""Targeted unit tests for check_code: finding shape, scoring, parsing fallbacks, and edge cases."""

from __future__ import annotations

import json

import pytest

from nba_auditor import DISCLAIMER, check_code
from nba_auditor.catalog import CHECKS
from nba_auditor.lint import classify, score_findings


def _rules(src: str) -> dict[str, str]:
    return {f.rule_id: f.severity for f in check_code(src).findings}


def test_finding_has_all_fields_and_explanations():
    a = check_code("import pandas as pd\ndf = pd.read_csv('p.csv')\ndf['signal'] = df['close'].shift(-1) > df['close']\n")
    f = a.findings[0]
    assert (f.rule_id, f.severity, f.line) == ("LA01", "high", 3)
    assert f.why == CHECKS["LA01"].why_it_inflates and f.fix == CHECKS["LA01"].how_to_fix
    assert "shift(-1)" in f.snippet
    d = a.to_dict()
    json.dumps(d)
    assert d["disclaimer"] == DISCLAIMER
    md = a.to_markdown()
    assert md.rstrip().endswith(DISCLAIMER)
    assert "LA01" in md and "line 3" in md


def test_every_rule_in_catalog_has_text():
    for c in CHECKS.values():
        assert c.what and c.why_it_inflates and c.how_to_fix and c.title


def test_score_is_transparent_and_capped_per_rule():
    src = "\n".join(["import pandas as pd", "df = pd.read_csv('x.csv')"] +
                    [f"df['sma{i}'] = df['close'].rolling({i + 5}, center=True).mean()" for i in range(5)])
    a = check_code(src)
    assert sum(f.rule_id == "LA02" for f in a.findings) == 5
    row = next(r for r in a.score_breakdown if r["rule_id"] == "LA02")
    assert row["penalty"] == 50 and row["findings"] == 5
    assert a.score == 50
    assert score_findings([])[0] == 100


def test_negative_shift_label_is_not_flagged():
    src = "import pandas as pd\ndf = pd.read_csv('x.csv')\ndf['fwd_ret'] = df['close'].shift(-5) / df['close'] - 1\n"
    assert _rules(src) == {}


def test_negative_shift_of_unknown_name_is_possible():
    src = "import pandas as pd\ndf = pd.read_csv('x.csv')\ndf['tmp'] = df['close'].shift(-1)\n"
    assert _rules(src) == {"LA01": "possible"}


def test_negative_shift_with_variable_horizon():
    src = "import pandas as pd\nh = 5\ndf = pd.read_csv('x.csv')\ndf['feature_a'] = df['close'].shift(-h)\n"
    assert _rules(src)["LA01"] == "high"


def test_negative_shift_in_if_condition():
    src = "import pandas as pd\ns = pd.Series([1.0, 2.0])\nif (s.shift(-1) > s).any():\n    print('x')\n"
    assert _rules(src)["LA01"] == "high"


def test_backfill_variants():
    base = "import pandas as pd\ndf = pd.read_csv('x.csv')\n"
    assert _rules(base + "df = df.backfill()\n")["LA03"] == "medium"
    assert _rules(base + "df = df.reindex(idx, method='bfill')\n")["LA03"] == "medium"
    assert _rules(base + "df = df.interpolate()\n")["LA03"] == "possible"
    assert "LA03" not in _rules(base + "df = df.ffill()\n")
    assert "LA03" not in _rules(base + "df = df.interpolate(method='pad')\n")


def test_merge_asof_nearest_and_backward():
    base = "import pandas as pd\n"
    assert _rules(base + "m = pd.merge_asof(a, b, on='date', direction='nearest')\n")["LA04"] == "medium"
    assert "LA04" not in _rules(base + "m = pd.merge_asof(a, b, on='date')\n")


def test_train_test_split_outside_time_series_context_is_only_possible():
    src = "from sklearn.model_selection import train_test_split\nX_tr, X_te = train_test_split(X, test_size=0.2)\n"
    assert _rules(src) == {"LA06": "possible"}


def test_fit_on_full_matrix_when_train_subset_exists():
    src = ("import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n"
           "df = pd.read_csv('x.csv', parse_dates=['date'])\nX = df[['a', 'b']]\n"
           "X_train = X.iloc[:500]\nscaler = StandardScaler()\nscaler.fit(X)\n")
    assert _rules(src)["LA05"] == "medium"


def test_global_minmax_scaling():
    src = ("import pandas as pd\ndf = pd.read_csv('x.csv', parse_dates=['date'])\n"
           "df['x_scaled'] = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())\n")
    assert _rules(src)["LA09"] == "medium"


def test_sharpe_formula_is_not_normalization():
    src = "import numpy as np\nr = load()\nsharpe = r.mean() / r.std() * np.sqrt(252)\nz = (r.mean() - 0) / r.std()\n"
    assert "LA09" not in _rules(src)


def test_grid_size_is_reported():
    src = ("import numpy as np\nimport itertools\nret = load()\nbest = -1\n"
           "for a, b in itertools.product([1, 2, 3], [10, 20]):\n"
           "    pnl = ret * a / b - 0.001 * abs(a)\n"
           "    sharpe = pnl.mean() / pnl.std()\n"
           "    if sharpe > best:\n        best = sharpe\n")
    f = next(x for x in check_code(src).findings if x.rule_id == "LA11")
    assert "about 6 combinations" in f.message and "n_trials>=6" in f.message


def test_bar_loop_is_not_a_grid():
    src = ("import pandas as pd\ndf = pd.read_csv('x.csv')\nbest_total = 0\ncost = 1\n"
           "for i in range(len(df)):\n    pnl = df['ret'][i] - cost\n    best_total = max(best_total, pnl)\n")
    assert "LA11" not in _rules(src)


def test_resample_daily_rules_not_flagged():
    base = "import pandas as pd\ns = load()\n"
    for rule in ("D", "W", "ME", "MS", "B"):
        assert "LA12" not in _rules(base + f"x = s.resample('{rule}').last()\n"), rule
    for rule in ("1h", "15T", "30s", "min"):
        assert _rules(base + f"x = s.resample('{rule}').last()\n").get("LA12") == "possible", rule


def test_today_constituent_helpers_and_csv_snapshot():
    assert _rules("from yahoo_fin import stock_info as si\nt = si.tickers_sp500()\n")["SB01"] == "high"
    assert _rules("import pandas as pd\nu = pd.read_csv('sp500_tickers.csv')\n")["SB01"] == "possible"
    assert "SB01" not in _rules("import pandas as pd\nu = pd.read_csv('sp500_membership_history.csv')\n")


def test_read_html_url_through_variable():
    src = ("import pandas as pd\nURL = 'https://en.wikipedia.org/wiki/Nasdaq-100'\n"
           "tickers = pd.read_html(URL)[4]['Ticker'].tolist()\n")
    assert _rules(src)["SB01"] == "high"


def test_signal_times_return_of_unknown_origin_is_medium():
    src = ("import pandas as pd\ndf = pd.read_csv('signals.csv')\nfee = 0.001\n"
           "df['pnl'] = df['signal'] * df['ret'] - fee * df['signal'].diff().abs()\n")
    assert _rules(src)["LA07"] == "medium"


def test_reassigning_signal_with_lag_clears_same_bar_rule():
    src = ("import pandas as pd\ndf = pd.read_csv('x.csv')\ndf['ret'] = df['close'].pct_change()\n"
           "df['signal'] = (df['close'] > df['close'].rolling(10).mean()).astype(int)\n"
           "df['signal'] = df['signal'].shift(1)\nfee = 0.001\n"
           "df['pnl'] = df['signal'] * df['ret'] - fee * df['signal'].diff().abs()\n")
    assert "LA07" not in _rules(src)


def test_trade_on_close_engine_setting():
    src = "from backtesting import Backtest\nbt = Backtest(data, Strat, commission=0.001, trade_on_close=True)\n"
    assert _rules(src)["LA07"] == "medium"


def test_backtesting_py_without_commission():
    src = "from backtesting import Backtest\nbt = Backtest(data, Strat)\n"
    assert _rules(src)["CO01"] == "possible"


def test_notebook_magics_are_ignored():
    src = "%matplotlib inline\n!pip install yfinance\nimport pandas as pd\ndf = pd.read_csv('x.csv')\ndf['sma'] = df['close'].rolling(5, center=True).mean()\n"
    a = check_code(src)
    assert a.parsed and any(f.rule_id == "LA02" for f in a.findings)
    assert any("magics" in n for n in a.notes)


def test_unparseable_code_falls_back_to_text_patterns():
    src = "def broken(:\n    x = df['close'].shift(-1)\n"
    a = check_code(src)
    assert not a.parsed
    assert [(f.rule_id, f.severity) for f in a.findings] == [("LA01", "possible")]


def test_empty_source_rejected():
    with pytest.raises(ValueError):
        check_code("   ")


@pytest.mark.parametrize("name,role", [
    ("fwd_ret_5d", "label"), ("target", "label"), ("y", "label"), ("y_pred", "signal"), ("Position", "signal"),
    ("strategy_returns", "pnl"), ("Returns", "return"), ("SMA20", "feature"), ("feat_next_ret", "feature"),
    ("hit_rate", "eval"), ("foo", "unknown"),
])
def test_name_roles(name, role):
    assert classify(name) == role

"""check_code v0.2: new and widened rules, measured on two corpora of our own.

* tests/corpus_v02: the builder's development set (written before and while building the rules, so in-sample).
* tests/corpus_v02_holdout: written by a separate agent that never saw the lint code, from a list of leak
  categories only. First measurement, before any fix that it prompted (2026-10-16):
  expected rule at high/medium 29/35, any high/medium 31/35; clean files with a high/medium finding 4/25
  (all four were v0.1 false alarms: LA10 on a metric, LA01 on a next-open fill price, LA07 through a lag helper,
  LA07 on a position carried from the previous loop iteration). v0.1 on the same set: 7/35, 11/35 and 5/25.
  Those fixes were then made, so the numbers asserted below are in-sample.

Each file starts with `# expect: RULE[:severity]` or `# expect: none`.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

from nba_auditor import check_code

HERE = Path(__file__).parent
DEV = HERE / "corpus_v02"
HOLD = HERE / "corpus_v02_holdout"
# The holdout author filed point-in-time fundamentals under LA04 (as-of merges); the lint reports them as LA15.
# Two numpy-offset files were filed under LA01; the lint reports the same leak as LA07/LA10.
ALIASES = {"LA04": {"LA04", "LA15"}, "LA01": {"LA01", "LA07", "LA10"}}


def _expect(path: Path) -> dict[str, str]:
    line = re.search(r"# expect: (.*)", path.read_text(encoding="utf-8")).group(1).strip()
    if line == "none":
        return {}
    return {i.strip().partition(":")[0]: i.strip().partition(":")[2] for i in line.split(",")}


def _hm(src: str):
    return [f for f in check_code(src).findings if f.severity in ("high", "medium")]


@pytest.mark.parametrize("path", sorted((DEV / "leaky").glob("*.py")) + sorted((HOLD / "leaky").glob("*.py")),
                         ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_leak_is_flagged_high_or_medium(path):
    exp = _expect(path)
    findings = check_code(path.read_text(encoding="utf-8")).findings
    hm = [f for f in findings if f.severity in ("high", "medium")]
    got = {(f.rule_id, f.severity) for f in hm}
    for rule, sev in exp.items():
        if sev == "possible":  # v0.3: CO01 (costs) is a caveat, capped at "possible"
            assert (rule, "possible") in {(f.rule_id, f.severity) for f in findings}, f"{path.name}: expected {rule}:possible"
            continue
        ok = {(r, s) for r, s in got if r in ALIASES.get(rule, {rule})}
        assert ok, f"{path.name}: expected {rule} at high/medium, got {sorted(got)}"
        if sev:
            assert sev in {s for _r, s in ok}, f"{path.name}: expected {rule}:{sev}, got {sorted(ok)}"


@pytest.mark.parametrize("path", sorted((DEV / "clean").glob("*.py")) + sorted((HOLD / "clean").glob("*.py")),
                         ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_clean_lookalike_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {[(f.rule_id, f.severity, f.line, f.message) for f in hm]}"


def test_corpora_are_balanced():
    for root in (DEV, HOLD):
        n_leaky, n_clean = len(list((root / "leaky").glob("*.py"))), len(list((root / "clean").glob("*.py")))
        assert n_leaky >= 30 and n_clean >= 25, (root.name, n_leaky, n_clean)


# ------------------------------------------------------------------ targeted cases
def _rules(src: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in check_code(src).findings:
        out.setdefault(f.rule_id, set()).add(f.severity)
    return out


def test_issue3_zscore_inside_function_taking_df():
    top = ('import pandas as pd\ndf = pd.read_csv("p.csv")\ndf["mom"] = df["close"].pct_change(20)\n'
           'df["z"] = (df["mom"] - df["mom"].mean()) / df["mom"].std()\ndf["signal"] = (df["z"] > 1).astype(int)\n')
    fn = ('import pandas as pd\ndef f(df):\n    df["mom"] = df["close"].pct_change(20)\n'
          '    df["z"] = (df["mom"] - df["mom"].mean()) / df["mom"].std()\n'
          '    df["signal"] = (df["z"] > 1).astype(int)\n    return df\n')
    assert "medium" in _rules(top)["LA09"]
    assert "medium" in _rules(fn)["LA09"]


def test_cross_section_function_is_not_full_sample():
    src = ("import pandas as pd\n"
           "def score(day):\n    day['z'] = (day['roe'] - day['roe'].mean()) / day['roe'].std()\n"
           "    day['long'] = day['z'] > 1\n    return day\n"
           "panel = pd.read_csv('p.csv')\nout = panel.groupby('date').apply(score)\n")
    assert "LA09" not in {r for r, s in _rules(src).items() if s & {"high", "medium"}}


def test_np_roll_direction():
    lead = "import numpy as np\nc = np.load('c.npy')\nsig = (np.roll(c, -1) > c).astype(int)\n"
    lag = "import numpy as np\nc = np.load('c.npy')\nsig = (np.roll(c, 1) < c).astype(int)\n"
    assert "high" in _rules(lead)["LA01"]
    assert "LA01" not in _rules(lag)


def test_positional_slices_same_bar_vs_aligned():
    base = "import numpy as np\nc = np.loadtxt('c.csv')\nr = c[1:] / c[:-1] - 1\nsig = np.where(c[1:] > c[:-1], 1, -1)\n"
    assert "high" in _rules(base + "pnl = sig * r - 0.001\n")["LA07"]
    assert "LA07" not in _rules(base + "pnl = sig[:-1] * r[1:] - 0.001\n")


def test_open_fill_needs_a_close_based_signal():
    head = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\n"
            "df['intraday'] = df['close'] / df['open'] - 1\n")
    leak = head + "df['sig'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)\ndf['pnl'] = df['sig'] * df['intraday'] - 0.0005\n"
    gap = head + "df['sig'] = (df['open'] < df['close'].shift(1)).astype(int)\ndf['pnl'] = df['sig'] * df['intraday'] - 0.0005\n"
    assert "high" in _rules(leak)["LA07"]
    assert "LA07" not in _rules(gap)


def test_early_stopping_on_test_vs_validation():
    src = ("from xgboost import XGBClassifier\nm = XGBClassifier(early_stopping_rounds=20)\n"
           "m.fit(X_train, y_train, eval_set=[({x}, {y})])\n")
    assert "high" in _rules(src.format(x="X_test", y="y_test"))["LA14"]
    assert "LA14" not in _rules(src.format(x="X_val", y="y_val"))
    monitor = "from xgboost import XGBClassifier\nm = XGBClassifier()\nm.fit(X_train, y_train, eval_set=[(X_test, y_test)])\n"
    assert "LA14" not in _rules(monitor)


def test_fundamentals_period_end_vs_filing_date():
    src = ("import pandas as pd\npx = pd.read_csv('p.csv')\nf = pd.read_csv('f.csv')\n"
           "m = pd.merge_asof(px, f, left_on='date', right_on='{k}', by='ticker')\n")
    assert "high" in _rules(src.format(k="period_end"))["LA15"]
    assert "LA15" not in _rules(src.format(k="filing_date"))


def test_resample_realigned_vs_labeled():
    src = ("import pandas as pd\nm1 = pd.read_csv('m.csv', index_col=0, parse_dates=True)\n"
           "h = m1['close'].resample('1h'{kw}).last()\nm1['h'] = h.reindex(m1.index, method='ffill')\n")
    assert "medium" in _rules(src.format(kw=""))["LA12"]
    assert "LA12" not in _rules(src.format(kw=", label='right', closed='right'"))


def test_costs_are_never_above_possible():
    base = ("import pandas as pd\ndf = pd.read_csv('spy.csv')\n"
            "df['position'] = (df['close'] > df['close'].rolling(50).mean()).astype(int).shift(1)\n"
            "df['strategy'] = df['position'] * df['close'].pct_change()\n")
    assert _rules(base)["CO01"] == {"possible"}
    # v0.3: capped at "possible" even when a performance figure is computed (a realism caveat, not lookahead)
    assert _rules(base + "print((1 + df['strategy']).prod())\n")["CO01"] == {"possible"}
    assert "CO01" not in _rules(base + "df['net'] = df['strategy'] - 0.0005 * df['position'].diff().abs()\n")
    # a comment saying "no commissions" is not a cost
    assert "CO01" in _rules("# no commissions here\n" + base)


def test_reversed_frame_then_rolling():
    src = ("import pandas as pd\ndf = pd.read_csv('p.csv')\ndf = df.sort_values('date', ascending=False)\n"
           "df['ma'] = df['close'].rolling(20).mean()\n")
    assert "high" in _rules(src)["LA02"]
    newest_first_fix = "import pandas as pd\ndf = pd.read_csv('p.csv')[::-1]\ndf['ma'] = df['close'].rolling(20).mean()\n"
    assert "LA02" not in _rules(newest_first_fix)


def test_self_referencing_comprehension_does_not_fall_back():
    src = "import pandas as pd\nx = [1, 2]\nx = [v for v in x]\nfor a in x:\n    best = max(a, 1)\n"
    assert check_code(src).parsed


def test_every_rule_id_used_in_lint_has_an_explain_entry():
    from nba_auditor.catalog import CHECKS

    lint_src = (HERE.parent / "nba_auditor" / "lint.py").read_text(encoding="utf-8")
    used = set(re.findall(r'self\.add\("([A-Z]{2}\d{2})"', lint_src))
    assert used <= set(CHECKS), used - set(CHECKS)
    assert {"LA14", "LA15"} <= used
    ast.parse(lint_src)


# ------------------------------------------------------------------ assignment targets (sealed-set false alarm)
_LHS = ("import numpy as np\nimport pandas as pd\nc = pd.read_csv('spy.csv')['close'].to_numpy()\n"
        "ret = np.zeros_like(c)\nret[1:] = c[1:] / c[:-1] - 1\n"
        "sig = c > pd.Series(c).rolling(20).mean().to_numpy()\npos = np.zeros_like(ret)\n")


def test_lhs_offset_slice_is_a_lag():
    """pos[1:] = f(sig[:-1]): the signal of bar t-1 sets the position of bar t. Correct, no high finding."""
    src = _LHS + "pos[1:] = np.where(sig[:-1], 1, -1)\npnl = pos * ret - 0.0005\n"
    assert not {s for s in _rules(src).get("LA07", set())} & {"high", "medium"}


def test_lhs_offset_slice_same_bar_is_still_a_leak():
    src = _LHS + "pos[1:] = np.where(sig[1:], 1, -1)\npnl = pos * ret - 0.0005\n"
    assert "high" in _rules(src)["LA07"]


def test_lhs_constant_lag_and_loop_element_lag():
    lag = ("LAG = 2\nimport numpy as np\nc = np.loadtxt('c.csv')\nr = c[1:] / c[:-1] - 1\nsig = np.sign(r)\n"
           "pos = np.zeros_like(r)\npos[LAG:] = sig[:-LAG]\npnl = pos * r - 0.001\n")
    assert "LA07" not in _rules(lag)
    loop = _LHS + "for i in range(len(c) - 1):\n    pos[i + 1] = 1.0 if sig[i] else 0.0\npnl = pos * ret - 0.0005\n"
    assert "LA07" not in _rules(loop)
    same = _LHS + "for i in range(len(c)):\n    pos[i] = 1.0 if sig[i] else 0.0\npnl = pos * ret - 0.0005\n"
    assert "high" in _rules(same)["LA07"]


def test_ambiguous_lhs_write_is_at_most_possible():
    """A mask or variable offset can't be lined up statically: never high on it."""
    for write in ("pos[sig] = 1.0\n", "k = int(input())\npos[k:] = sig[:-k]\n"):
        got = _rules(_LHS + write + "pnl = pos * ret - 0.0005\n").get("LA07", set())
        assert not got & {"high", "medium"}, (write, got)


def test_survivor_filter_on_last_date():
    leak = ("import pandas as pd\np = pd.read_csv('p.csv')\nlast = p.groupby('ticker')['date'].max()\n"
            "keep = last[last == p['date'].max()].index\np = p[p['ticker'].isin(keep)]\n")
    report = ("import pandas as pd\np = pd.read_csv('p.csv')\nlast = p.groupby('ticker')['date'].max()\n"
              "print((last < p['date'].max()).sum())\n")
    assert "medium" in _rules(leak)["SB01"]
    assert "SB01" not in _rules(report)


# ------------------------------------------------------------------ red-team review of PR #8 (tests/corpus_rt3)
RT3 = HERE / "corpus_rt3"
# leaks the red team wrote that v0.2 flags at high/medium; the rest are documented misses (README, Known limitations)
RT3_CAUGHT = {"l01_xgb_val_then_test_last", "l02_lgb_valid_sets_val_test", "l04_keras_es_val_test_named_oos",
              "l10_merge_asof_index", "l11_survivor_groupby_filter", "l13_survivor_isin_last_day",
              "l16_resample_on_col_merge", "l18_sort_ascending_zero", "l19_percentile_threshold", "l23_parametergrid",
              "l24_slice_same_bar_np_diff_sign", "l25_np_r_same_bar", "l26_survivor_transform_last_eq",
              "l31_catboost_val_test", "m01_df_col_rank", "m04_zscore_series_var"}


@pytest.mark.parametrize("path", sorted((RT3 / "clean").glob("*.py")), ids=lambda p: p.stem)
def test_rt3_correct_code_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {[(f.rule_id, f.severity, f.line, f.message) for f in hm]}"


def test_rt3_caught_leaks_stay_caught():
    caught = {p.stem for p in (RT3 / "leaky").glob("*.py") if _hm(p.read_text(encoding="utf-8"))}
    second_round = {p.stem for p in (RT3 / "leaky").glob("r[345]_*.py")}  # all later-round leaks are caught
    assert RT3_CAUGHT | second_round <= caught, sorted((RT3_CAUGHT | second_round) - caught)


def test_pandas_shift_before_to_numpy_is_a_lag():
    src = ("import numpy as np, pandas as pd\nc = np.load('c.npy')\nsig = (c > np.roll(c, 1)).astype(float)\n"
           "ret = np.diff(c) / c[:-1]\npos = pd.Series(sig).shift(1).fillna(0).to_numpy()\npnl = pos[1:] * ret - 0.0005\n")
    assert not _rules(src).get("LA07", set()) & {"high", "medium"}
    same = src.replace(".shift(1).fillna(0)", ".fillna(0)")
    assert "high" in _rules(same)["LA07"]


@pytest.mark.parametrize("lag", ["np.r_[0.0, sig[:-1]]", "np.concatenate(([0.0], sig[:-1]))", "np.hstack(([0.0], sig[:-1]))",
                                 "np.pad(sig[:-1], (1, 0))", "np.pad(sig, (1, 0))[:-1]", "np.insert(sig[:-1], 0, 0.0)",
                                 "np.append(0.0, sig[:-1])"])
def test_full_length_lag_idioms(lag):
    head = ("import numpy as np\nc = np.load('c.npy')\nsig = (c > np.roll(c, 1)).astype(float)\n"
            "ret = np.r_[np.nan, c[1:] / c[:-1] - 1]\n")
    assert not _rules(head + f"pos = {lag}\npnl = pos * ret - 0.0005\n").get("LA07", set()) & {"high", "medium"}
    same = lag.replace("sig[:-1]", "sig[1:]") if "sig[:-1]" in lag else lag.replace("(1, 0))[:-1]", "(0, 0))")
    assert "high" in _rules(head + f"pos = {same}\npnl = pos * ret - 0.0005\n").get("LA07", set()), same


# Timing budgets. The product requirement is the web check's 8 s timeout: a 200 KB input must never reach it, on
# any machine, so that ceiling is always asserted. The tighter engineering budget (3 s) catches quadratic scans on a
# developer machine; shared CI runners vary by +-15% or more (PR #11 saw 3.07-3.38 s with no lint change), so CI
# gets 5 s.
HARD_CEILING_S = 8.0
BUDGET_S = 5.0 if os.environ.get("CI") else 3.0


def _timed(src: str) -> float:
    """Seconds to lint `src`. Near the budget, the median of three runs (a busy machine adds noise to single runs;
    a quadratic scan doesn't go away). Every run must stay under the hard ceiling."""
    import time

    def once() -> float:
        t = time.perf_counter()
        check_code(src)
        dt = time.perf_counter() - t
        assert dt < HARD_CEILING_S, f"{dt:.2f}s reaches the web check's {HARD_CEILING_S:.0f}s timeout"
        return dt

    first = once()
    if first < BUDGET_S * 0.8:
        return first
    return sorted([first, once(), once()])[1]


def _assert_fast(label: str, dt: float) -> None:
    assert dt < HARD_CEILING_S, f"{label}: {dt:.2f}s reaches the web check's {HARD_CEILING_S:.0f}s timeout"
    assert dt < BUDGET_S, f"{label}: {dt:.2f}s over the {BUDGET_S:.0f}s budget ({'CI' if os.environ.get('CI') else 'local'})"


def test_large_input_is_fast():
    """A ~200 KB file of realistic snippets must lint in a few seconds (no quadratic scans)."""
    srcs = [p.read_text(encoding="utf-8") for p in sorted(HERE.glob("corpus*/*/*.py"))]
    blob, i = [], 0
    while sum(map(len, blob)) < 200_000:
        blob.append(srcs[i % len(srcs)])
        i += 1
    _assert_fast("200 KB corpus blob", _timed("\n".join(blob)))
    # many grid loops over a name bound to a list (used to rescan the whole tree per loop)
    grid = "g = [5, 10, 20]\n" + "".join(f"s{k} = []\nfor p in g:\n    s{k}.append(sharpe(df, p))\nbest{k} = max(s{k})\n"
                                         for k in range(2600))
    _assert_fast("grid loops", _timed(grid))


def test_heavy_dataflow_inputs_are_fast():
    """Red-team stress shapes (PR #9 rounds 5-6): long chains of forward columns, outcomes and lags."""
    hdr = ('import pandas as pd, numpy as np\ndf["fwd_ret"]=df["close"].pct_change().shift(-1)\n'
           'df["signal"]=np.sign(df["close"].diff())\n')
    s1 = hdr + "".join(f'df["e{k}"]=df["signal"]*df["fwd_ret"] + df["e{k - 1 if k else 0}"]*0.5 + '
                       f'(df["fwd_ret"].groupby(df["date"]).mean()*df["signal"]).sum()\n' for k in range(1500))
    s3 = hdr + "".join(f'v{k} = df["fwd_ret"].shift(-{k % 7})\nw{k} = v{k}.shift(1)\nposition{k} = np.sign(w{k})\n'
                       for k in range(2000))
    _assert_fast("s1 outcome chain", _timed(s1))
    _assert_fast("s3 lead/lag chain", _timed(s3))


def test_period_end_lag_rules():
    base = ("import pandas as pd\npx = pd.read_csv('p.csv', parse_dates=['date'])\nf = pd.read_csv('f.csv')\n{lag}"
            "m = pd.merge_asof(px, f, left_on='date', right_on='period_end'{kw})\n")

    def flagged(lag, kw=""):
        return "LA15" in _rules(base.format(lag=lag, kw=kw))

    assert flagged("", ", tolerance=pd.Timedelta(days=120)")  # a tolerance is not a reporting lag
    assert flagged("TOL = pd.Timedelta(days=120)\n", ", tolerance=TOL")
    assert flagged("f['period_end'] = f['period_end'] + pd.offsets.MonthEnd(0)\n")  # a snap, not a lag
    assert not flagged("f['period_end'] = f['period_end'] + pd.Timedelta(days=90)\n")
    assert not flagged("f['period_end'] = f['period_end'].add(pd.DateOffset(months=3))\n")
    assert not flagged("LAG = pd.Timedelta(days=75)\nf['period_end'] = f['period_end'] + LAG\n")

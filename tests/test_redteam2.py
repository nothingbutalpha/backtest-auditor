"""Regression tests for the second red-team pass (held-out corpus in tests/corpus_rt2, verdict holes, robustness).

The corpus_rt2 snippets were written by the red team without looking at the lint's rules. Files listed in
KNOWN_MISSES are still missed and are documented in README "Known limitations"; the recall floor below
keeps the measured held-out number from silently regressing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_auditor import InputError, audit_returns, check_code
from nba_auditor.returns_audit import VERDICT_NO_FLAGS

CORPUS = Path(__file__).parent / "corpus_rt2"
LEAKY = sorted((CORPUS / "leaky").glob("*.py"))
CLEAN = sorted((CORPUS / "clean").glob("*.py"))
IDX = pd.bdate_range("2005-01-03", periods=3000)

# file stem -> rule that must fire at high or medium
EXPECTED = {
    "h03_helper_fn": "LA01",
    "h04_listcomp_future": "LA13",
    "h05_iloc_next_in_loop_assign": "LA13",
    "h06_loc_future_date": "LA13",
    "h07_reversed_rolling_max": "LA02",
    "h09_groupby_transform_z_full": "LA09",
    "h10_groupby_transform_mean": "LA09",
    "h21_quantile_threshold_var": "LA09",
    "h22_minmax_full_np": "LA09",
    "h26_reversed_expanding_min": "LA02",
    "h28_np_where_same_bar": "LA07",
    "h29_open_to_close_same_day": "LA07",
    "h30_scaler_all_then_slice": "LA05",
    "h32_grid_dict_full_sample": "LA11",
    "m02_while_loop_next": "LA13",
    "m08_dict_comprehension_future": "LA13",
    # caught since v0.2 (positional alignment, full-sample qcut, vectorbt price=close, abs-max scaling)
    "h01_np_slice": "LA07",
    "h11_qcut_full": "LA09",
    "h16_vbt_price_same_close": "LA07",
    "m03_zip_shifted_lists": "LA07",
    "m04_max_full_sample_normalize_signal": "LA09",
}
KNOWN_MISSES: set[str] = set()  # v0.1 missed 5 of these 44; all are regression tests now, so this is in-sample


def _hm(src: str):
    return [f for f in check_code(src).findings if f.severity in ("high", "medium")]


@pytest.mark.parametrize("stem,rule", sorted(EXPECTED.items()))
def test_heldout_leak_is_flagged(stem, rule):
    src = (CORPUS / "leaky" / f"{stem}.py").read_text(encoding="utf-8")
    rules = {f.rule_id for f in _hm(src)}
    assert rule in rules, f"{stem}: expected {rule} at high/medium, got {sorted(rules)}"


@pytest.mark.parametrize("path", CLEAN, ids=lambda p: p.stem)
def test_heldout_clean_has_no_high_or_medium(path):
    hm = _hm(path.read_text(encoding="utf-8"))
    assert not hm, f"{path.name}: false alarms {[(f.rule_id, f.line, f.message) for f in hm]}"


def test_heldout_recall_floor():
    caught = [p.stem for p in LEAKY if _hm(p.read_text(encoding="utf-8"))]
    missed = sorted({p.stem for p in LEAKY} - set(caught))
    assert set(missed) <= KNOWN_MISSES, missed
    assert len(caught) / len(LEAKY) >= 1.0


def test_vbt_from_signals_with_explicit_close_price_is_noted():
    src = (CORPUS / "leaky" / "h16_vbt_price_same_close.py").read_text(encoding="utf-8")
    assert any(f.rule_id == "LA07" for f in check_code(src).findings)


# ------------------------------------------------------------------ robustness
def test_deeply_nested_expression_does_not_crash():
    a = check_code("x = " + "+".join(["a"] * 5000) + "\nsig = df['c'].shift(-1) > 0\n")
    assert a.parsed is False
    assert any(f.rule_id == "LA01" for f in a.findings)


def test_oversized_source_is_rejected():
    with pytest.raises(ValueError, match="too large"):
        check_code("x = 1\n" * 400_000)


def test_mcp_rejects_oversized_inputs():
    from nba_auditor.mcp import audit_returns as mcp_audit_returns, check_code as mcp_check_code

    assert "error" in mcp_check_code("x = 1\n" * 400_000)
    big = "date,return\n" + "2020-01-01,0.01\n" * 2_000_000
    assert "error" in mcp_audit_returns(big)


def test_mcp_recursion_error_is_reported_not_raised():
    from nba_auditor.mcp import check_code as mcp_check_code

    out = mcp_check_code("x = " + "+".join(["a"] * 5000))
    assert "markdown" in out or "error" in out


# ------------------------------------------------------------------ audit_returns holes
def test_infinite_returns_are_rejected():
    x = np.r_[np.random.default_rng(1).normal(0.001, 0.01, 300), np.inf]
    with pytest.raises(InputError, match="finite"):
        audit_returns(pd.Series(x, index=IDX[:301]), n_trials=1, turnover=0.01)


def _best_of(seed, k=100, T=1260):
    r = np.random.default_rng(10_000 + seed).normal(0, 0.01, (T, k))
    return pd.Series(r[:, np.argmax(r.mean(0) / r.std(0))], index=IDX[:T])


def test_understated_trials_sr_variance_cannot_give_no_red_flags():
    """Best of 100 noise strategies, honest n_trials=100, but a tiny trials_sr_variance used to pass ~50%."""
    verdicts = [audit_returns(_best_of(s), n_trials=100, turnover=0.05, trials_sr_variance=0.01).verdict
                for s in range(40)]
    assert VERDICT_NO_FLAGS not in verdicts
    a = audit_returns(_best_of(0), n_trials=100, turnover=0.05, trials_sr_variance=0.01)
    assert any("trials_sr_variance" in w for w in a.warnings)


def test_assumed_frequency_cannot_give_no_red_flags():
    x = list(np.random.default_rng(5).normal(0.0015, 0.01, 1000))  # strong drift, no dates
    a = audit_returns(x, n_trials=1, turnover=0.05)
    assert a.verdict != VERDICT_NO_FLAGS
    assert any("frequency" in r.lower() for r in a.reasons)
    # an explicit frequency is the user's statement and is accepted
    bench = list(np.random.default_rng(6).normal(0.0003, 0.01, 1000))  # independent market, same length
    assert audit_returns(x, freq="D", n_trials=1, turnover=0.05, benchmark=bench).verdict == VERDICT_NO_FLAGS


def test_mixed_frequencies_are_flagged():
    d1 = pd.bdate_range("2010-01-01", periods=600)
    d2 = pd.date_range(d1[-1] + pd.Timedelta(days=31), periods=200, freq="ME")
    r = np.r_[np.random.default_rng(2).normal(0.002, 0.01, 600), np.random.default_rng(3).normal(0.03, 0.04, 200)]
    a = audit_returns(pd.Series(r, index=d1.append(d2)), n_trials=1, turnover=0.05)
    assert a.verdict != VERDICT_NO_FLAGS
    assert any("irregular" in w.lower() for w in a.warnings)


def test_regular_daily_with_holidays_is_not_irregular():
    idx = pd.bdate_range("2010-01-01", periods=1500)
    idx = idx[(idx.month != 12) | (idx.day != 25)]
    a = audit_returns(pd.Series(np.random.default_rng(6).normal(0.0009, 0.01, len(idx)), index=idx),
                      n_trials=1, turnover=0.05)
    assert not any("irregular" in w.lower() for w in a.warnings)

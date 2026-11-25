"""Regression tests for the red-team review of v0.1 (must-fix items 1-7 and 10, plus should-fix 12-14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_auditor import InputError, audit_returns, audit_trade_log
from nba_auditor.returns_audit import VERDICT_MAJOR_FLAGS, VERDICT_NO_FLAGS
from nba_auditor.trade_log import parse_trades

IDX = pd.bdate_range("2010-01-04", periods=4000)


def test_overlapping_returns_from_noise_are_never_no_red_flags():
    """Item 1: 20-day overlapping sums of noise (autocorrelation ~0.95) used to fool the iid DSR."""
    reals = 0
    for seed in range(60):
        e = np.random.default_rng(seed).normal(0, 0.01, 1279)
        x = np.convolve(e, np.ones(20), "valid") / 20
        a = audit_returns(pd.Series(x, index=IDX[:len(x)]), n_trials=10, turnover=0.05)
        reals += a.verdict == VERDICT_NO_FLAGS
        if seed == 0:
            assert a.to_dict()["performance"]["variance_ratio"] > 5
            assert any("autocorrelated" in w for w in a.warnings)
    assert reals == 0


def test_hac_bandwidth_sees_overlap():
    from nba_auditor.stats import hac_bandwidth, newey_west_lags

    e = np.random.default_rng(1).normal(0, 1, 1279)
    x = np.convolve(e, np.ones(20), "valid")
    assert hac_bandwidth(x) >= 30 > newey_west_lags(len(x))
    assert hac_bandwidth(np.random.default_rng(2).normal(0, 1, 1260)) <= 10


def test_benchmark_exposure_is_not_no_red_flags():
    """Item 2: a 2x leveraged market against the market has zero alpha."""
    m = np.random.default_rng(3).normal(0.1 / 252, 0.01, 3780)
    for strat in (m, 2 * m):
        a = audit_returns(pd.Series(strat, index=IDX[:3780]), n_trials=1, turnover=0.0,
                          benchmark=pd.Series(m, index=IDX[:3780]))
        assert a.verdict != VERDICT_NO_FLAGS
        assert any("No significant alpha" in r for r in a.reasons)


NO_BENCH = "No benchmark supplied: market exposure can't be separated from skill."


def test_no_benchmark_caps_verdict_at_some_red_flags():
    """CEO decision: without a benchmark the best possible verdict is 'Some red flags'."""
    planted = pd.Series(np.random.default_rng(11).normal(0.0009, 0.01, 2520), index=IDX[:2520])
    a = audit_returns(planted, n_trials=10, turnover=0.05)
    assert a.verdict == "Some red flags"
    assert NO_BENCH in a.reasons
    market = pd.Series(np.random.default_rng(99).normal(0.0003, 0.01, 2520), index=IDX[:2520])
    b = audit_returns(planted, n_trials=10, turnover=0.05, benchmark=market)
    assert b.verdict == VERDICT_NO_FLAGS and NO_BENCH not in b.reasons


def test_leveraged_index_without_benchmark_never_gets_no_red_flags():
    """A 2x leveraged index with a strong drift looks great against zero; it must never pass."""
    for seed in range(20):
        idx_ret = np.random.default_rng(200 + seed).normal(0.12 / 252, 0.01, 3780)  # Sharpe ~0.75 index
        lev = pd.Series(2 * idx_ret, index=IDX[:3780])
        for kw in ({"turnover": 0.0}, {"turnover": 0.01, "costs_bps": 2}, {"returns_are_net": True}):
            a = audit_returns(lev, n_trials=1, **kw)
            assert a.verdict != VERDICT_NO_FLAGS
            if a.verdict == "Some red flags":
                assert NO_BENCH in a.reasons
    a = audit_returns(lev, n_trials=1, turnover=0.0)
    assert NO_BENCH in a.reasons or a.verdict == "Major red flags"


def test_monthly_returns_default_to_inferred_frequency():
    """Item 3: monthly data used to be read as daily and called implausible."""
    m = pd.Series(np.random.default_rng(4).normal(0.012, 0.035, 240),
                  index=pd.date_range("2005-01-31", periods=240, freq="ME"))
    a = audit_returns(m, n_trials=1, turnover=0.05)
    assert any("12 periods per year" in x for x in a.assumptions)
    assert "implausibility" not in a.to_dict()
    with pytest.raises(InputError, match="spaced like M"):
        audit_returns(m, freq="D")


def test_intraday_series_cannot_be_no_red_flags():
    """Item 4."""
    idx = pd.date_range("2024-01-02 09:30", periods=3000, freq="h")
    x = pd.Series(np.random.default_rng(5).normal(0.0004, 0.002, 3000), index=idx)
    a = audit_returns(x, freq=252 * 6.5, n_trials=1, turnover=0.01)
    assert a.verdict != VERDICT_NO_FLAGS
    assert any("Intraday" in r for r in a.reasons)
    with pytest.raises(InputError, match="intraday"):
        audit_returns(x)


def test_cost_rule_uses_the_declared_cost():
    """Item 5: gross 8 bp/day, turnover 1, declared cost 1 bp is not 'costs erase it'."""
    r = pd.Series(np.random.default_rng(6).normal(0.0008, 0.01, 1260), index=IDX[:1260])
    a = audit_returns(r, turnover=1.0, costs_bps=1, n_trials=5)
    assert not any("Costs erase it" in x for x in a.reasons)
    b = audit_returns(r, turnover=1.0, costs_bps=10, n_trials=5)
    assert b.verdict == VERDICT_MAJOR_FLAGS and any("Costs erase it: at 10 bps" in x for x in b.reasons)


def test_percent_units_block_no_red_flags():
    """Should-fix 14."""
    r = pd.Series(np.random.default_rng(7).normal(0.5, 0.3, 1260), index=IDX[:1260])
    a = audit_returns(r, n_trials=1, turnover=0.01)
    assert a.verdict != VERDICT_NO_FLAGS


def _split_case(adjusted: bool):
    days = pd.bdate_range("2021-01-04", periods=60)
    raw = np.r_[np.full(30, 400.0), np.full(30, 100.0)] * np.exp(np.random.default_rng(8).normal(0, 0.005, 60))
    marks = raw.copy()
    if adjusted:
        marks[:30] = marks[:30] / 4
    trades = pd.DataFrame([dict(date=days[5].date(), ticker="NV", side="buy", qty=10, price=raw[5]),
                           dict(date=days[50].date(), ticker="NV", side="sell", qty=10, price=raw[50])])
    prices = pd.DataFrame({"date": days, "NV": marks})
    return trades.to_csv(index=False), prices.to_csv(index=False)


def test_adjusted_marks_vs_raw_fills_are_caught():
    """Item 6 (the 4:1 split case is simplified: quantities are not split-adjusted here)."""
    t0, p0 = _split_case(adjusted=False)
    assert not any("Fill prices differ" in w for w in audit_trade_log(t0, prices=p0, initial_capital=100_000).warnings)
    t, p = _split_case(adjusted=True)
    a = audit_trade_log(t, prices=p, initial_capital=100_000)
    assert any("Fill prices differ" in w for w in a.warnings)
    assert a.verdict != VERDICT_NO_FLAGS


def test_trade_log_side_and_fee_ambiguity():
    """Item 7."""
    with pytest.raises(InputError, match="no side column"):
        parse_trades("date,symbol,quantity,price\n2024-01-02,A,10,5\n2024-01-03,A,10,6\n")
    t = parse_trades("date,symbol,transaction_type,quantity,price,cost\n2024-01-02,A,BUY,10,5,50\n2024-01-03,A,SELL,10,6,60\n")
    assert list(t["qty"]) == [10.0, -10.0] and (t["fee"] == 0).all()  # 'cost' is not taken as a fee
    with pytest.raises(InputError, match="notional"):
        parse_trades("date,ticker,side,qty,price,fee\n2024-01-02,A,buy,10,5,50\n2024-01-03,A,sell,10,6,60\n")
    t = parse_trades("date,ticker,side,qty,price\n2024-01-02,A,sell to open,10,5\n2024-01-03,A,buy_to_close,10,4\n")
    assert list(t["qty"]) == [-10.0, 10.0]


def test_reused_ticker_message_is_hedged():
    """Item 10: ARM, SNOW, CEG and META were used by earlier companies."""
    days = pd.bdate_range("2005-03-01", periods=60)
    trades = pd.DataFrame([dict(date=days[0].date(), ticker="ARM", side="buy", qty=1, price=10.0),
                           dict(date=days[45].date(), ticker="ARM", side="sell", qty=1, price=10.0)])
    a = audit_trade_log(trades.to_csv(index=False), prices=pd.DataFrame({"date": days, "ARM": 10.0}).to_csv(index=False))
    msg = a.to_dict()["survivorship_hints"]["hints"][0]["message"]
    assert "company that uses this symbol today" in msg and "earlier company" in msg
    assert a.verdict != VERDICT_NO_FLAGS


def test_flat_book_is_not_padded_with_zero_days():
    """Should-fix 12: a price file that runs past the last trade."""
    days = pd.bdate_range("2020-01-02", periods=300)
    px = pd.DataFrame({"date": days, "A": np.linspace(10, 20, 300)})
    trades = "date,ticker,side,qty,price\n" + "".join(
        f"{days[i].date()},A,{'buy' if i % 2 == 0 else 'sell'},10,{px['A'][i]:.4f}\n" for i in range(0, 100))
    a = audit_trade_log(trades, prices=px.to_csv(index=False))
    assert a.to_dict()["performance"]["n"] <= 100

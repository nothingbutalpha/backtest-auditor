"""audit_trade_log: accounting, survivorship hints, and the no-prices safeguard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_auditor import InputError, audit_trade_log
from nba_auditor.returns_audit import VERDICT_NO_FLAGS
from nba_auditor.trade_log import build_daily, parse_prices, parse_trades


def test_mark_to_market_hand_example():
    trades = parse_trades("date,ticker,side,qty,price\n2024-01-02,ABC,buy,100,10.00\n2024-01-04,ABC,sell,100,11.20\n")
    prices = parse_prices("date,ABC\n2024-01-02,10.50\n2024-01-03,11.00\n2024-01-04,11.00\n2024-01-05,11.50\n")
    d = build_daily(trades, prices, initial_capital=10_000)
    pnl = d["gross_ret"] * 10_000  # equity_prev is 10_000 on day 1 only; recompute absolute PnL below
    eq = 10_000.0
    absolute = []
    for r in d["gross_ret"]:
        absolute.append(r * eq)
        eq += r * eq
    assert absolute[:3] == pytest.approx([50.0, 50.0, 20.0])  # 100*(10.5-10), 100*(11-10.5), -100*(11-11.2)
    assert sum(absolute) == pytest.approx(120.0)  # = 100 * (11.20 - 10.00)
    assert d["turnover"].iloc[0] == pytest.approx(1000 / 10_000)
    assert pnl.iloc[0] == pytest.approx(50.0)


def test_short_and_cover_and_signed_qty():
    a = parse_trades("date,symbol,action,shares,fill_price\n2024-01-02,XYZ,short,50,20\n2024-01-03,XYZ,cover,50,18\n")
    b = parse_trades("date,ticker,qty,price\n2024-01-02,XYZ,-50,20\n2024-01-03,XYZ,50,18\n")
    assert list(a["qty"]) == list(b["qty"]) == [-50.0, 50.0]
    d = build_daily(a, None, initial_capital=1000)
    assert (d["gross_ret"] * 1000).sum() == pytest.approx(100.0, rel=1e-6)


def test_bad_trade_logs():
    with pytest.raises(InputError, match="ticker"):
        parse_trades("date,qty,price\n2024-01-02,1,2\n")
    with pytest.raises(InputError, match="side"):
        parse_trades("date,ticker,side,qty,price\n2024-01-02,A,hold,1,2\n")


def _synthetic(seed=3, tickers=("AAA", "BBB", "CCC", "DDD"), start="2016-01-04", end="2021-12-31", drift=0.0002):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    px = pd.DataFrame({t: 50 * np.exp(np.cumsum(rng.normal(drift, 0.015, len(days)))) for t in tickers}, index=days)
    rows = []
    for i in range(0, len(days) - 5, 5):
        t = tickers[(i // 5) % len(tickers)]
        rows.append(dict(date=days[i].date(), ticker=t, side="buy", qty=100, price=round(px.at[days[i], t], 2), fee=1.0))
        rows.append(dict(date=days[i + 5].date(), ticker=t, side="sell", qty=100, price=round(px.at[days[i + 5], t], 2), fee=1.0))
    return pd.DataFrame(rows).to_csv(index=False), px.reset_index(names="date").to_csv(index=False)


def test_anachronistic_ticker_hint():
    trades, prices = _synthetic(tickers=("AAA", "BBB", "CCC", "META"))
    a = audit_trade_log(trades, prices=prices, n_trials=1)
    hints = a.to_dict()["survivorship_hints"]
    assert hints["heuristic"] is True
    assert [h["ticker"] for h in hints["hints"] if h["type"] == "anachronistic_ticker"] == ["META"]
    assert any("TL01" in w for w in a.warnings)


def test_no_departures_hint_for_many_names():
    names = tuple(f"T{i:02d}" for i in range(25))
    trades, prices = _synthetic(tickers=names, start="2015-01-02", end="2020-12-31")
    a = audit_trade_log(trades, prices=prices)
    types = [h["type"] for h in a.to_dict()["survivorship_hints"]["hints"]]
    assert "no_disappearing_names" in types


def test_without_prices_cannot_be_no_red_flags():
    trades, _ = _synthetic(drift=0.003)  # strongly trending: big, smooth gains
    a = audit_trade_log(trades, n_trials=1)
    assert a.verdict != VERDICT_NO_FLAGS
    assert any("No daily prices" in r for r in a.reasons)


def test_fees_in_log_are_net_and_sections_present():
    trades, prices = _synthetic()
    a = audit_trade_log(trades, prices=prices)
    d = a.to_dict()
    assert d["tool"] == "audit_trade_log" and d["costs"]["returns_are_net"] is True
    assert d["trade_log"]["fees_total"] == pytest.approx(624.0)
    assert len(d["ticker_concentration"]["top5"]) == 4
    md = a.to_markdown()
    assert "## Trade log" in md and "Survivorship hints" in md


def test_same_close_leak_trade_log_is_flagged():
    """Buy at today's close only on days the stock goes up (decided with today's close): a fill on the
    signal bar. The returns look implausible and the audit must say so."""
    rng = np.random.default_rng(21)
    days = pd.bdate_range("2019-01-02", periods=500)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 500))), index=days)
    rows = []
    for i in range(1, 499):
        if px.iloc[i + 1] > px.iloc[i]:  # peeks at the next close
            rows.append(dict(date=days[i].date(), ticker="SPYX", side="buy", qty=100, price=px.iloc[i]))
            rows.append(dict(date=days[i + 1].date(), ticker="SPYX", side="sell", qty=100, price=px.iloc[i + 1]))
    trades = pd.DataFrame(rows).to_csv(index=False)
    prices = px.rename("SPYX").rename_axis("date").reset_index().to_csv(index=False)
    a = audit_trade_log(trades, prices=prices, n_trials=1, costs_bps=1)
    assert a.verdict == "Major red flags"
    assert any("Implausible" in r for r in a.reasons)

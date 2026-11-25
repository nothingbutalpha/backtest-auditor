"""`audit_trade_log`: turn a trade log into daily PnL and turnover, then run `audit_returns`.

Accounting
----------
Positions are marked to market every day:

    PnL_t = sum_i pos_{i,t-1} * (P_{i,t} - P_{i,t-1})           (holding PnL)
          + sum_{trades on t} q * (P_{i,t} - fill_price)        (trade PnL vs the day's mark)
          - fees_t

P is the closing price from `prices` if you supply one, otherwise the last fill price seen for that
ticker (so positions are only re-marked when they trade). Returns are PnL_t / equity_{t-1}, with
equity starting at `initial_capital` (default: the largest gross exposure in the log, at fill prices).
Turnover_t = traded notional_t / equity_{t-1}.

Without a price file the equity curve is stale between trades, so volatility and drawdowns are
understated and the Sharpe ratio overstated. The verdict therefore can't be "No red flags found" without prices.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .parsing import InputError, read_table
from .returns_audit import ReturnsAudit, audit_returns

BUY_WORDS = {"buy", "b", "long", "cover", "buy_to_cover", "buytocover", "btc", "bto", "buy_to_open", "buy to cover",
             "buy to open", "bot", "bought", "buy_to_close", "buy to close"}
SELL_WORDS = {"sell", "s", "short", "sell_short", "sellshort", "ss", "sto", "stc", "sell_to_open", "sell_to_close",
              "sell short", "sell to open", "sell to close", "sld", "sold"}

# Symbols that did not exist (as a listed US ticker) before these dates. Trading one of them earlier
# means today's symbol was mapped back onto history. Public listing / symbol-change dates.
SYMBOL_FIRST_DATE = {
    "META": "2022-06-09",  # Facebook renamed; traded as FB before
    "GOOGL": "2014-04-03",  # class A got GOOGL after the stock split; Google traded as GOOG before
    "BKNG": "2018-02-27",  # Priceline traded as PCLN before
    "ABBV": "2013-01-02",
    "PYPL": "2015-07-20",
    "KHC": "2015-07-06",
    "CARR": "2020-04-03",
    "OTIS": "2020-04-03",
    "CEG": "2022-02-02",
    "GEHC": "2023-01-04",
    "KVUE": "2023-05-04",
    "GEV": "2024-04-02",
    "SOLV": "2024-04-01",
    "UBER": "2019-05-10",
    "LYFT": "2019-03-29",
    "ABNB": "2020-12-10",
    "DASH": "2020-12-09",
    "SNOW": "2020-09-16",
    "PLTR": "2020-09-30",
    "COIN": "2021-04-14",
    "RIVN": "2021-11-10",
    "ARM": "2023-09-14",
    "FB": "2012-05-18",
    "TSLA": "2010-06-29",
}


def _col(df: pd.DataFrame, names: tuple[str, ...], required: bool, what: str) -> str | None:
    lower = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    if required:
        raise InputError(f"trade log: missing a {what} column (looked for {', '.join(names)}); "
                         f"columns are {[str(c) for c in df.columns]}")
    return None


MAX_TICKER_CHARS = 16
_TICKER_BAD = re.compile(r"[^A-Z0-9.\-_/^=:#]")


def clean_ticker(s: Any) -> str:
    """Tickers are echoed in reports: keep them short and plain (letters, digits, . - _ / ^ = : #), so a trade log
    can't smuggle long text or markup into a report through its ticker column."""
    t = _TICKER_BAD.sub("?", str(s).strip().upper())
    return t if len(t) <= MAX_TICKER_CHARS else t[:MAX_TICKER_CHARS - 1] + "~"


def parse_trades(trades: Any) -> pd.DataFrame:
    df = read_table(trades, "trade log")
    if df.empty:
        raise InputError("trade log: no rows")
    dcol = _col(df, ("date", "datetime", "time", "timestamp", "trade_date", "filled_at", "exec_time"), True, "date")
    tcol = _col(df, ("ticker", "symbol", "asset", "instrument", "security"), True, "ticker")
    qcol = _col(df, ("qty", "quantity", "shares", "size", "amount", "units", "filled_qty"), True, "quantity")
    pcol = _col(df, ("price", "fill_price", "avg_price", "exec_price", "execution_price", "fill", "trade_price"), True, "price")
    scol = _col(df, ("side", "action", "direction", "type", "buy_sell", "b/s", "transaction_type", "trade_type",
                     "order_side", "transaction", "activity"), False, "side")
    # 'cost' is deliberately not a fee alias: broker exports often use it for the trade's notional.
    fcol = _col(df, ("fee", "fees", "commission", "commissions", "fees_and_commissions"), False, "fee")
    try:
        dates = pd.to_datetime(df[dcol], format="mixed")
    except (ValueError, TypeError):
        raise InputError(f"trade log: could not parse dates in column {dcol!r}") from None
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    out = pd.DataFrame({
        "date": dates,
        "ticker": df[tcol].map(clean_ticker),
        "qty": pd.to_numeric(df[qcol], errors="coerce"),
        "price": pd.to_numeric(df[pcol], errors="coerce"),
        "fee": pd.to_numeric(df[fcol], errors="coerce").fillna(0.0).abs() if fcol else 0.0,
    })
    if out[["qty", "price"]].isna().any().any():
        bad = int(out[["qty", "price"]].isna().any(axis=1).sum())
        raise InputError(f"trade log: {bad} row(s) have a non-numeric quantity or price")
    if (out["price"] <= 0).any():
        raise InputError("trade log: prices must be positive")
    if scol is not None:
        side = df[scol].astype(str).str.strip().str.lower()
        sign = np.where(side.isin(BUY_WORDS), 1.0, np.where(side.isin(SELL_WORDS), -1.0, np.nan))
        if np.isnan(sign).any():
            unknown = sorted(set(side[np.isnan(sign)]))[:5]
            raise InputError(f"trade log: unknown side value(s) {unknown}; use buy/sell (or short/cover)")
        out["qty"] = out["qty"].abs() * sign
    elif (out["qty"] > 0).all():
        raise InputError("trade log: no side column was recognised and every quantity is positive, so buys and sells "
                         "can't be told apart. Add a 'side' column (buy/sell/short/cover) or sign the quantities.")
    if fcol is not None:
        notional = (out["qty"].abs() * out["price"]).replace(0, np.nan)
        ratio = float((out["fee"] / notional).median())
        if ratio > 0.01:
            raise InputError(f"trade log: the fee column {fcol!r} is a median {ratio:.1%} of each trade's notional, "
                             "which looks like a notional or cost-basis column rather than fees. Rename or drop it.")
    out["date"] = out["date"].dt.normalize()
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def parse_prices(prices: Any) -> pd.DataFrame:
    """Daily closes, wide (date + one column per ticker) or long (date, ticker, close)."""
    df = read_table(prices, "prices")
    dcol = _col(df, ("date", "datetime", "time", "timestamp"), True, "date")
    tcol = _col(df, ("ticker", "symbol", "asset"), False, "ticker")
    ccol = _col(df, ("close", "adj_close", "price", "px", "last"), False, "close")
    df[dcol] = pd.to_datetime(df[dcol], format="mixed").dt.normalize()
    if tcol is not None and ccol is not None:
        df[tcol] = df[tcol].map(clean_ticker)
        wide = df.pivot_table(index=dcol, columns=tcol, values=ccol, aggfunc="last")
    else:
        wide = df.set_index(dcol).apply(pd.to_numeric, errors="coerce")
        wide.columns = [clean_ticker(c) for c in wide.columns]
    return wide.sort_index()


def build_daily(trades: pd.DataFrame, prices: pd.DataFrame | None, initial_capital: float | None) -> dict:
    tickers = sorted(trades["ticker"].unique())
    start, end = trades["date"].min(), trades["date"].max()
    if prices is not None:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise InputError(f"prices: no price column for {missing[:10]}")
        days = prices.index[(prices.index >= start)]
        days = days.union(pd.DatetimeIndex(trades["date"].unique())).sort_values()
        marks = prices.reindex(days)[tickers].ffill()
    else:
        days = pd.bdate_range(start, end).union(pd.DatetimeIndex(trades["date"].unique())).sort_values()
        last_fill = trades.groupby(["date", "ticker"])["price"].last().unstack()
        marks = last_fill.reindex(days).ffill()
    # a ticker with no mark yet on a day it trades gets that trade's fill price
    first_fill = trades.groupby("ticker")["price"].first()
    marks = marks.fillna(first_fill)
    fill_gap = None
    if prices is not None:
        m_at = np.array([marks.at[d, t] for d, t in zip(trades["date"], trades["ticker"])], float)
        fill_gap = float(np.nanmedian(np.abs(trades["price"].to_numpy() / m_at - 1)))

    q_day = trades.pivot_table(index="date", columns="ticker", values="qty", aggfunc="sum").reindex(days).fillna(0.0)
    q_day = q_day.reindex(columns=tickers, fill_value=0.0)
    notional = (trades.assign(n=trades["qty"].abs() * trades["price"])
                .groupby("date")["n"].sum().reindex(days).fillna(0.0))
    cash_flow = (trades.assign(c=trades["qty"] * trades["price"]).groupby(["date", "ticker"])["c"].sum()
                 .unstack().reindex(index=days, columns=tickers).fillna(0.0))
    fees = trades.groupby("date")["fee"].sum().reindex(days).fillna(0.0)

    pos = q_day.cumsum()
    pos_prev = pos.shift(1).fillna(0.0)
    m = marks[tickers]
    holding = (pos_prev * (m - m.shift(1))).fillna(0.0)
    trade_pnl = q_day * m - cash_flow  # q * (mark - fill), summed per ticker-day
    pnl_by_ticker = holding + trade_pnl
    gross_pnl = pnl_by_ticker.sum(axis=1)
    exposure = (pos.abs() * m).sum(axis=1)

    cap = float(initial_capital) if initial_capital else float(exposure.max())
    if cap <= 0:
        raise InputError("trade log: could not infer capital (no exposure); pass initial_capital")
    net_pnl = gross_pnl - fees
    equity_net = cap + net_pnl.cumsum()
    equity_prev = equity_net.shift(1).fillna(cap)
    if (equity_prev <= 0).any():
        raise InputError("trade log: equity hits zero or below with this capital; pass a larger initial_capital")
    gross_ret = gross_pnl / equity_prev
    turnover = notional / equity_prev
    final = pos.iloc[-1]
    if prices is not None and (final.abs() < 1e-9).all():
        keep = days <= end  # the book is flat after the last trade: don't pad with zero-return days
        days, gross_ret, turnover, fees, equity_prev = (days[keep], gross_ret[keep], turnover[keep], fees[keep],
                                                        equity_prev[keep])
    return {"days": days, "gross_ret": gross_ret, "fees_ret": fees / equity_prev, "turnover": turnover,
            "pnl_by_ticker": pnl_by_ticker.sum(axis=0) - trades.groupby("ticker")["fee"].sum().reindex(tickers).fillna(0.0),
            "capital": cap, "capital_inferred": not initial_capital, "final_positions": pos.iloc[-1],
            "net_pnl_total": float(net_pnl.sum()), "fees_total": float(fees.sum()), "fill_vs_mark_median": fill_gap}


def survivorship_hints(trades: pd.DataFrame) -> list[dict]:
    """Heuristic hints that the universe was built from today's names. Never proof."""
    hints = []
    span_days = (trades["date"].max() - trades["date"].min()).days
    tickers = trades["ticker"].unique()
    first = trades.groupby("ticker")["date"].min()
    last = trades.groupby("ticker")["date"].max()
    for t, d in SYMBOL_FIRST_DATE.items():
        if t in first.index and first[t] < pd.Timestamp(d):
            hints.append({"type": "anachronistic_ticker", "ticker": t, "first_trade": str(first[t].date()),
                          "symbol_first_traded": d,
                          "message": f"{t} is traded on {first[t].date()}, but the company that uses this symbol today "
                                     f"only started trading under it on {d}. Unless your log means an earlier company "
                                     "that used the same symbol, today's ticker was mapped back onto older history."})
    if span_days >= 3 * 365 and len(tickers) >= 20:
        cutoff = trades["date"].max() - pd.Timedelta(days=int(span_days * 0.2))
        early_exit = int((last < cutoff).sum())
        if early_exit == 0:
            hints.append({"type": "no_disappearing_names", "tickers": int(len(tickers)), "span_years": round(span_days / 365.25, 1),
                          "message": f"All {len(tickers)} tickers still trade in the last 20% of a {span_days / 365.25:.1f}-year log. "
                                     "Over several years some companies normally get acquired or delisted; a universe with "
                                     "no departures is often built from today's survivors."})
    return hints


def audit_trade_log(
    trades: Any,
    prices: Any = None,
    initial_capital: float | None = None,
    costs_bps: float | None = None,
    benchmark: Any = None,
    n_trials: int | None = None,
) -> ReturnsAudit:
    """Audit a trade log (date, ticker, side, qty, price[, fee]). Optionally pass daily closes as `prices`."""
    t = parse_trades(trades)
    px = parse_prices(prices) if prices is not None else None
    d = build_daily(t, px, initial_capital)
    warnings, assumptions, blockers = [], [], []
    has_fees = bool((t["fee"] > 0).any())
    gap = d.get("fill_vs_mark_median")
    if gap is not None and gap > 0.05:
        msg = (f"Fill prices differ from your daily closes by a median {gap:.0%} on trade days. The price file is "
               "probably split/dividend-adjusted while the fills are raw (or vice versa), which corrupts daily returns.")
        warnings.append(msg)
        blockers.append(msg)
    if px is None:
        blockers.append("No daily prices were given, so positions are only marked when they trade: volatility and "
                        "drawdowns are understated and the Sharpe ratio is overstated. Pass `prices` for a real audit.")
    if d["capital_inferred"]:
        assumptions.append(f"Capital inferred as the largest gross exposure in the log: {d['capital']:,.0f}. "
                           "Pass initial_capital if you started with a different amount.")
    open_pos = d["final_positions"][d["final_positions"].abs() > 1e-9]
    if len(open_pos):
        assumptions.append(f"{len(open_pos)} position(s) are still open at the end and are marked at the last price.")
    if has_fees:
        assumptions.append("Fees in the log are deducted in the headline result; the cost grid shows extra costs on top.")
        returns = d["gross_ret"] - d["fees_ret"]
    else:
        returns = d["gross_ret"]

    pnl = d["pnl_by_ticker"].sort_values(ascending=False)
    total = float(pnl.sum())
    top5 = pnl.head(5)
    ticker_conc = {
        "tickers": int(len(pnl)), "total_pnl": total,
        "top5": [{"ticker": k, "pnl": float(v)} for k, v in top5.items()],
        "top5_share_of_pnl": float(top5.sum() / total) if total > 0 else None,
        "pnl_without_top5": float(total - top5.sum()),
    }
    if total > 0 and total - top5.sum() <= 0 and len(pnl) > 5:
        warnings.append("Dropping the 5 best tickers turns total PnL negative (TL02): the result depends on a few names.")
    hints = survivorship_hints(t)
    for h in hints:
        warnings.append("Survivorship hint (heuristic, TL01): " + h["message"])
    if any(h["type"] == "anachronistic_ticker" for h in hints):
        blockers.append("A ticker is traded before its current company used the symbol (TL01): check the ticker mapping "
                        "and the universe for survivorship bias.")
    log_summary = {"trades": int(len(t)), "tickers": int(t["ticker"].nunique()),
                   "first_trade": str(t["date"].min().date()), "last_trade": str(t["date"].max().date()),
                   "days": int(len(d["days"])), "capital": d["capital"], "fees_total": d["fees_total"],
                   "net_pnl_total": d["net_pnl_total"], "marked_with_prices": px is not None}
    extra = {"trade_log": log_summary, "ticker_concentration": ticker_conc,
             "survivorship_hints": {"heuristic": True, "hints": hints,
                                    "note": "These are heuristics from the log alone (no external data). "
                                            "No hint does not mean no survivorship bias."}}
    s = pd.Series(returns.to_numpy(), index=d["days"])
    to = pd.Series(d["turnover"].to_numpy(), index=d["days"])
    return audit_returns(s, freq="D", costs_bps=costs_bps if costs_bps is not None else (0.0 if has_fees else None),
                         turnover=to, benchmark=benchmark, n_trials=n_trials, returns_are_net=has_fees,
                         _context={"warnings": warnings, "assumptions": assumptions, "blockers": blockers,
                                   "extra_sections": extra, "tool": "audit_trade_log"})

# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"])
f = pd.read_csv("fund.csv", parse_dates=["period_end"])
m = prices.merge(f, left_on=["ticker", "date"], right_on=["ticker", "period_end"], how="left")
m["eps"] = m.groupby("ticker")["eps"].transform("ffill")
m["signal"] = m["eps"] / m["close"]

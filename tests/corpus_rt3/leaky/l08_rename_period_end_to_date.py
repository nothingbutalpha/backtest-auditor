# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"])
fund = pd.read_csv("fund.csv", parse_dates=["period_end"]).rename(columns={"period_end": "date"})
df = prices.merge(fund, on=["ticker", "date"], how="left")
df["eps"] = df.groupby("ticker")["eps"].ffill()
df["signal"] = df["eps"] / df["close"]

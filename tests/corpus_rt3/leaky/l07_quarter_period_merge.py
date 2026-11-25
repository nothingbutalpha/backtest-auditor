# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"])
fund = pd.read_csv("fund.csv")
prices["quarter"] = prices["date"].dt.to_period("Q")
fund["quarter"] = pd.PeriodIndex(fund["fiscal_quarter"], freq="Q")
df = prices.merge(fund, on=["ticker", "quarter"], how="left")
df["signal"] = df["eps"] / df["close"]

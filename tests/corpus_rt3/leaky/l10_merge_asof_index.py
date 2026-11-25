# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", index_col="date", parse_dates=True).sort_index()
fund = pd.read_csv("fund.csv", parse_dates=["period_end"]).set_index("period_end").sort_index()
df = pd.merge_asof(prices, fund, left_index=True, right_index=True)
df["signal"] = df["eps"] / df["close"]

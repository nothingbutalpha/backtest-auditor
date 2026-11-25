# expect: LA03:medium, LA15:medium
# Quarterly fundamentals are joined to daily prices and back-filled: each day gets the NEXT report.
import pandas as pd

prices = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
fund = pd.read_csv("fundamentals.csv", parse_dates=["period_end"], index_col="period_end")
merged = prices.join(fund, how="left").bfill()
merged["eps_yield"] = merged["eps"] / merged["close"]
merged["book"] = merged["book"].fillna(method="bfill")

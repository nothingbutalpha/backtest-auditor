# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
MAX_STALE = pd.Timedelta(days=200)
prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
f = pd.read_csv("fund.csv", parse_dates=["datadate"]).sort_values("datadate")
m = pd.merge_asof(prices, f, left_on="date", right_on="datadate", by="gvkey", tolerance=MAX_STALE)

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
f = pd.read_csv("compustat.csv", parse_dates=["datadate"])
f["datadate"] = f["datadate"].apply(lambda d: d + pd.DateOffset(months=4))
f = f.sort_values("datadate")
m = pd.merge_asof(prices, f, left_on="date", right_on="datadate", by="gvkey")

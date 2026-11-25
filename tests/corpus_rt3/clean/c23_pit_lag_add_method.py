# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
REPORT_LAG = pd.Timedelta(days=75)
prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
f = pd.read_csv("fund.csv", parse_dates=["period_end"])
f["period_end"] = f["period_end"].add(REPORT_LAG)
f = f.sort_values("period_end")
m = pd.merge_asof(prices, f, left_on="date", right_on="period_end", by="ticker")

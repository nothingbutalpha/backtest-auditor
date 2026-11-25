# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"])
f = pd.read_csv("compustat.csv", parse_dates=["datadate", "rdq"])
m = prices.merge(f, left_on=["gvkey", "date"], right_on=["gvkey", "rdq"], how="left")
m["atq"] = m.groupby("gvkey")["atq"].ffill()

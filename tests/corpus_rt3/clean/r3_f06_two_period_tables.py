# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
inc = pd.read_csv("income.csv", parse_dates=["period_end"])
bal = pd.read_csv("balance.csv", parse_dates=["period_end"])
f = inc.merge(bal, on=["ticker", "period_end"])
f["roe"] = f["net_income"] / f["equity"]
f["available"] = f["period_end"] + pd.Timedelta(days=90)
prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
m = pd.merge_asof(prices, f.sort_values("available"), left_on="date", right_on="available", by="ticker")

# expect: none
import pandas as pd
inc = pd.read_csv("income.csv", parse_dates=["period_end", "filed"])
bal = pd.read_csv("balance.csv", parse_dates=["period_end"])
fund = inc.merge(bal, left_on=["ticker", "period_end"], right_on=["ticker", "period_end"])
fund = fund.sort_values("filed")
px = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
df = pd.merge_asof(px, fund, left_on="date", right_on="filed", by="ticker")
df["roe"] = df["net_income"] / df["equity"]

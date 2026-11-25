# expect: none
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
fund = pd.read_csv("income_statements.csv", parse_dates=["period_end", "filing_date"]).sort_values("filing_date")
df = pd.merge_asof(prices, fund, left_on="date", right_on="filing_date", by="ticker")
df["ep"] = df["eps_ttm"] / df["close"]
df["signal"] = (df["ep"] > 0.06).astype(int)

# expect: LA15:high
import pandas as pd
prices = pd.read_csv("prices.csv", parse_dates=["date"]).sort_values("date")
fund = pd.read_csv("income_statements.csv", parse_dates=["period_end"]).sort_values("period_end")
df = pd.merge_asof(prices, fund, left_on="date", right_on="period_end", by="ticker")
df["ep"] = df["eps_ttm"] / df["close"]
df["signal"] = (df["ep"] > 0.06).astype(int)

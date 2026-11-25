import pandas as pd
px = pd.read_csv("px.csv", parse_dates=["date"])
fund = pd.read_csv("fundamentals.csv", parse_dates=["period_end"])
df = px.merge(fund, left_on="date", right_on="period_end", how="left")
df["eps"] = df["eps"].bfill()
df["signal"] = (df["eps"] > 0).astype(int)

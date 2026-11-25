import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["future_min"] = df["Close"].iloc[::-1].expanding().min().iloc[::-1]
df["position"] = (df["Close"] <= df["future_min"] * 1.01).astype(int)
df["strat"] = df["position"].shift(1) * df["Close"].pct_change()

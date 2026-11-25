import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["Close"].pct_change(20)
df["bucket"] = pd.qcut(df["mom"], 5, labels=False)
df["position"] = (df["bucket"] == 4).astype(int)
df["strat"] = df["position"].shift(1) * df["Close"].pct_change()

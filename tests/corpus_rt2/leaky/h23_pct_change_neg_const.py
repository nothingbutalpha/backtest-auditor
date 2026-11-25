import pandas as pd
HORIZON = 5
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["Close"].pct_change(periods=-HORIZON)
df["signal"] = (df["mom"] > 0).astype(int)
df["strat"] = df["signal"].shift(1) * df["Close"].pct_change()

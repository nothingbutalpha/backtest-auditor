import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["vol"] = df["ret"].rolling(20).std()
df["vol_next"] = df["vol"].shift(-1)
df["weight"] = 0.01 / df["vol_next"]
df["strat"] = df["weight"].shift(1) * df["ret"]

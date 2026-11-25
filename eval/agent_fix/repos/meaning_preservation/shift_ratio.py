import pandas as pd

df = pd.read_csv("spy.csv", index_col=0, parse_dates=True).sort_index()
df["signal"] = (df["close"].shift(-1) / df["close"] - 1 > 0).astype(int)
df["position"] = df["signal"].shift(1)
df["strat"] = df["position"] * df["close"].pct_change() - 0.0005

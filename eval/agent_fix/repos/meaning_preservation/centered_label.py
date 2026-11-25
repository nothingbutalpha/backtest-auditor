import pandas as pd

df = pd.read_csv("spy.csv", index_col=0, parse_dates=True).sort_index()
df["trend"] = df["close"].rolling(21, center=True).mean()
df["label"] = (df["trend"].diff() > 0).astype(int)
df["mom"] = df["close"].pct_change(20)

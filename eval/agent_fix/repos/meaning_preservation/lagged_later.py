import pandas as pd

df = pd.read_csv("spy.csv", index_col=0, parse_dates=True).sort_index()
df["fast"] = df["close"].rolling(10).mean()
df["slow"] = df["close"].rolling(50).mean()
df["signal"] = (df["fast"].shift(-1) > df["slow"]).astype(int)
df["position"] = df["signal"].shift(1)

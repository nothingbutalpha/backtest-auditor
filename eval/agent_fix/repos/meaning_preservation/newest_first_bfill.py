import pandas as pd

df = pd.read_csv("newest_first.csv", index_col=0, parse_dates=True)
df["close"] = df["close"].bfill()
df = df.sort_index()
df["signal"] = df["close"] > df["close"].rolling(20).mean()

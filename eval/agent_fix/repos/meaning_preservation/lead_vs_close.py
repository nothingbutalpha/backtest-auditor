import pandas as pd

df = pd.read_csv("spy.csv", index_col=0, parse_dates=True).sort_index()
df["nxt"] = df["close"].shift(-1)
df["signal"] = (df["nxt"] > df["close"]).astype(int)

import pandas as pd

df = pd.read_csv("spy.csv", index_col=0, parse_dates=True).sort_index()
df["mom"] = df["close"].pct_change(-10)
df["chg"] = df["close"].diff(-5)
df["signal"] = ((df["mom"] > 0) & (df["chg"] > 0)).astype(int)

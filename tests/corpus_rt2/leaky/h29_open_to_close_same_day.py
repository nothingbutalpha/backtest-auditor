import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["intraday"] = df["Close"] / df["Open"] - 1
df["signal"] = (df["Close"] > df["Close"].rolling(5).mean()).astype(int)
df["strat"] = df["signal"] * df["intraday"]

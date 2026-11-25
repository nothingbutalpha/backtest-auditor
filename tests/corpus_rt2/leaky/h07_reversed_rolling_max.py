import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["peak_ahead"] = df["Close"][::-1].rolling(20).max()[::-1]
df["signal"] = (df["Close"] < df["peak_ahead"] * 0.95).astype(int)
df["strat"] = df["signal"].shift(1) * df["Close"].pct_change()

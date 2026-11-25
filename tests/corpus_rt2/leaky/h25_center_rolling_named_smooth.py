import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["smooth"] = df["Close"].rolling(11, center=True).mean()
df["signal"] = (df["Close"] > df["smooth"]).astype(int)
df["strat"] = df["signal"].shift(1) * df["Close"].pct_change()

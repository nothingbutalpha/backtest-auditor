import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
wk = df["Close"].resample("W").last()
df["week_close"] = wk.reindex(df.index, method="bfill")
df["signal"] = (df["Close"] < df["week_close"]).astype(int)
df["strat"] = df["signal"].shift(1) * df["Close"].pct_change()

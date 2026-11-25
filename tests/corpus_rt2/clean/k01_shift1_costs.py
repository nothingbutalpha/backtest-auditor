import pandas as pd, numpy as np
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["sma"] = df["Close"].rolling(50).mean()
df["signal"] = (df["Close"] > df["sma"]).astype(int)
df["strat"] = df["signal"].shift(1) * df["ret"] - df["signal"].diff().abs() * 0.0005
print(df["strat"].mean() / df["strat"].std() * np.sqrt(252))

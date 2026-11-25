# expect: none
import numpy as np
import pandas as pd

h = 5
df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["y"] = np.where(df["close"].shift(-h) > df["close"], 1, 0)
df["fwd_return"] = df["close"].pct_change(h).shift(-h)
df["ma_gap"] = df["close"] / df["close"].rolling(50).mean() - 1
df = df.iloc[:-h]

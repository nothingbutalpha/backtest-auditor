# expect: LA10:high
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["signal"] = np.where(df["fwd_ret"].rolling(3).mean() > 0, 1, -1)
cost = 0.0005
df["strategy"] = df["signal"] * df["fwd_ret"] - cost * df["signal"].diff().abs()

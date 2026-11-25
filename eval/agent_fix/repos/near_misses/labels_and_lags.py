import numpy as np
import pandas as pd

COST = 0.0005

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["ret"] = df["close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df["target"] = (df["fwd_ret"] > 0).astype(int)
df["sma"] = df["close"].rolling(20, center=False).mean()
df["vol"] = df["ret"].rolling(20).std()
df["signal"] = (df["close"] > df["sma"]).astype(int)
df["strategy"] = df["signal"] * df["fwd_ret"] - df["signal"].diff().abs() * COST
print("hit rate", (np.sign(df["strategy"]) > 0).mean())

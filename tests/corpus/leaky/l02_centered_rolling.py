# expect: LA02:high
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["sma"] = df["close"].rolling(20, center=True).mean()
df["position"] = np.where(df["close"] > df["sma"], 1, -1)
df["position"] = df["position"].shift(1)
df["ret"] = df["close"].pct_change()
cost_bps = 5
df["strategy"] = df["position"] * df["ret"] - df["position"].diff().abs() * cost_bps / 1e4

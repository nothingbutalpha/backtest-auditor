# expect: none
# Looks like a helper-function shift leak, but the helper lags with s.shift(n) for positive n.
import numpy as np
import pandas as pd


def lag(s, n=1):
    return s.shift(n)


df = pd.read_csv("qqq.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["ema"] = df["close"].ewm(span=10).mean()
df["signal"] = np.where(lag(df["close"]) > lag(df["ema"]), 1, 0)
df["position"] = lag(df["signal"]).fillna(0)
df["cost"] = df["position"].diff().abs().fillna(0) * 0.0005
df["strategy"] = df["position"] * df["ret"] - df["cost"]
print("Total return:", (1 + df["strategy"].fillna(0)).prod() - 1)

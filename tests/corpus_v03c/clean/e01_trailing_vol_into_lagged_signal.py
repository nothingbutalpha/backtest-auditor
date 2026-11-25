# expect: none
# Twin of leaky/e01: trailing realized vol (known at t).
import numpy as np
import pandas as pd

df = pd.read_csv("spy_iv.csv", parse_dates=["date"], index_col="date")
r = df["close"].pct_change()
df["rv"] = r.rolling(21).std() * np.sqrt(252)
df["vrp"] = df["iv"] - df["rv"]
df["sig"] = (df["vrp"] > 0.02).astype(int).shift(1)
df["pnl"] = df["sig"] * r - 0.0005 * df["sig"].diff().abs()

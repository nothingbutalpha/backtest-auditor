# expect: LA01|LA10:high
# Realized vol of the NEXT 21 days enters a spread; the signal is lagged by only 1 day.
import numpy as np
import pandas as pd

df = pd.read_csv("spy_iv.csv", parse_dates=["date"], index_col="date")
r = df["close"].pct_change()
df["rv"] = r.rolling(21).std().shift(-21) * np.sqrt(252)
df["vrp"] = df["iv"] - df["rv"]
df["sig"] = (df["vrp"] > 0.02).astype(int).shift(1)
df["pnl"] = df["sig"] * r - 0.0005 * df["sig"].diff().abs()

# expect: LA10:high
# A rolling mean of the forward return used as a signal (no lag).
import numpy as np
import pandas as pd

df = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["signal"] = np.where(df["fwd_ret"].rolling(5).mean() > 0, 1, -1)
df["strategy"] = df["signal"] * df["fwd_ret"] - 0.0005 * df["signal"].diff().abs()

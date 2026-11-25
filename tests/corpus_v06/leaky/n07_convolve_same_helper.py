# expect: LA02:high@7
import numpy as np
import pandas as pd


def centered_mean(x, n=9):
    return np.convolve(x, np.ones(n) / n, mode="same")


px = pd.read_parquet("silver.parquet")["close"]
signal = pd.Series(np.where(px.to_numpy() > centered_mean(px.to_numpy()), 1.0, -1.0), index=px.index)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())

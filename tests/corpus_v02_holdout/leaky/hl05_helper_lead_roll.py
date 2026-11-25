# expect: LA01
# Utility `lead` wraps np.roll(a, -k), so the volume/breakout features look k bars ahead.
import numpy as np
import pandas as pd


def lead(a, k=1):
    return np.roll(a, -k)


bars = pd.read_parquet("es_5min.parquet")
px = bars["close"].values
vol = bars["volume"].values
ret = np.r_[0.0, px[1:] / px[:-1] - 1]

vol_spike = lead(vol, 3) > 2 * vol
breakout = lead(px, 3) > px
sig = (vol_spike & breakout).astype(float)
pos = np.r_[0.0, sig[:-1]]
net = pos * ret - 0.0002 * np.abs(np.diff(pos, prepend=0.0))
print("net pnl", net.sum())

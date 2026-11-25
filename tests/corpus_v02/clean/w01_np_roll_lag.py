# expect: none
import numpy as np
import pandas as pd
px = pd.read_csv("spy.csv")["close"].to_numpy()
prev = np.roll(px, 1)
prev[0] = np.nan
signal = np.where(px > prev, 1, 0)
ret = np.r_[np.nan, px[1:] / px[:-1] - 1]
pnl = signal[:-1] * ret[1:] - 0.0005 * np.abs(np.diff(signal))
print(np.nanmean(pnl))

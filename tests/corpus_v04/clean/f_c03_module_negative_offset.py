# expect: none
import numpy as np

LOOKBACK = -3
c = np.load("closes.npy")
sig = np.zeros(len(c))
for i in range(3, len(c)):
    if c[i + LOOKBACK] < c[i] * 0.97:
        sig[i] = 1
ret = np.r_[0.0, np.diff(c) / c[:-1]]
pnl = sig[:-1] * ret[1:]
print(pnl.sum())

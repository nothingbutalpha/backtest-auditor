# expect: none
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
c = pd.read_csv("btc_1h.csv")["close"].to_numpy()
sig = np.zeros(len(c))
for t in range(100, len(c)):
    pk, _ = find_peaks(c[t - 100:t], distance=12)
    if len(pk) and pk[-1] == 99 - 3:
        sig[t] = -1
pnl = sig[:-1] * np.diff(c) / c[:-1]
print(pnl.sum())

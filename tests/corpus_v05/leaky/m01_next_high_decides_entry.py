# expect: LA13:high
import numpy as np
import pandas as pd

d = pd.read_csv("spy_daily.csv")
o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = d["close"].rolling(20).mean().to_numpy()
trades = []
for i in range(20, len(c) - 2):
    if h[i + 1] > c[i] * 1.02:
        entry_px = o[i + 1]
        trades.append(c[i + 1] / entry_px - 1)
print(np.mean(trades))

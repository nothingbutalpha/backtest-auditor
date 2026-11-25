# expect: none
import numpy as np
import pandas as pd

d = pd.read_csv("spy_daily.csv")
o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = d["close"].rolling(20).mean().to_numpy()
out = []
for i in range(20, len(c) - 6):
    limit = c[i] * 0.985
    if c[i] < sma[i] * 0.97 and lo[i + 1] <= limit:
        buy_px = min(limit, o[i + 1])
        sell_px = c[i + 6]
        out.append(sell_px / buy_px - 1 - 0.001)
print(np.mean(out))

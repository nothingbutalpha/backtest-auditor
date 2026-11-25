# expect: none
import numpy as np
import pandas as pd

d = pd.read_csv("qqq.csv")
o, lo, c = (d[k].to_numpy() for k in ("open", "low", "close"))
sma = d["close"].rolling(30).mean().to_numpy()
HOLD = 4
trades = []
for i in range(30, len(c) - HOLD - 1):
    if len(trades) >= 100:
        break
    if c[i] < sma[i]:
        limit = c[i] * 0.995
        if lo[i + 1] <= limit:
            fill = min(limit, o[i + 1])
            trades.append(c[i + 1 + HOLD] / fill - 1)
print(len(trades), np.mean(trades))

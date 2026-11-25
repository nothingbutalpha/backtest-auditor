# expect: LA13:high
import numpy as np
import pandas as pd

d = pd.read_csv("spy_daily.csv")
o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = d["close"].rolling(20).mean().to_numpy()
pnl = []
for i in range(20, len(c) - 6):
    if c[i] < sma[i]:
        limit = c[i] * 0.99
        if lo[i + 1] <= limit:
            fill = min(limit, o[i + 1])
            size = (h[i + 1] - lo[i + 1]) / c[i] * 10
            pnl.append(size * (c[i + 6] / fill - 1))
print(sum(pnl))

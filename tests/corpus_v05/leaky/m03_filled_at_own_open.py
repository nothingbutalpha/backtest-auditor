# expect: LA07|LA13:high
import numpy as np
import pandas as pd

d = pd.read_csv("spy_daily.csv")
o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = d["close"].rolling(20).mean().to_numpy()
pnl = []
for i in range(20, len(c) - 6):
    if c[i] < sma[i]:
        entry_px = o[i]
        pnl.append(c[i + 5] / entry_px - 1)
print(sum(pnl))

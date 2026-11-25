# expect: none
"""Signal at the close, entry at the next open, take-profit or stop resting during that bar, else exit at its close."""
import numpy as np
import pandas as pd

d = pd.read_csv("spy_daily.csv")
o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = d["close"].rolling(20).mean().to_numpy()
pnl = []
for i in range(20, len(c) - 1):
    if c[i] > sma[i] and c[i - 1] <= sma[i - 1]:
        entry_px = o[i + 1]
        tp = entry_px * 1.02
        stop = entry_px * 0.99
        if h[i + 1] >= tp:
            exit_px = tp
        elif lo[i + 1] <= stop:
            exit_px = stop
        else:
            exit_px = c[i + 1]
        pnl.append(exit_px / entry_px - 1 - 0.0005)
print(sum(pnl))

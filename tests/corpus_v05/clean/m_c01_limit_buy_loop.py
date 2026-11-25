# expect: none
"""Buy-limit 1% under the close when the close is below the 20-day mean; the order rests during the next bar."""
import numpy as np
import pandas as pd

d = pd.read_csv("spy_daily.csv")
o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = d["close"].rolling(20).mean().to_numpy()
HOLD = 5
trades = []
for i in range(20, len(c) - HOLD - 1):
    if c[i] < sma[i]:
        limit = c[i] * 0.99
        if lo[i + 1] <= limit:
            fill = min(limit, o[i + 1])
            exit_px = c[i + 1 + HOLD]
            trades.append(exit_px / fill - 1 - 0.001)
print(len(trades), np.mean(trades))

# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("es_daily.csv", parse_dates=["date"])
c = df["close"].to_numpy()
o = df["open"].to_numpy()
hi = df["high"].rolling(20).max().shift(1).to_numpy()
pos, entry_px, pnl = 0, 0.0, []
for t in range(21, len(df) - 1):
    go_long = c[t] > hi[t]
    if go_long and pos == 0:
        pos = 1
        entry_px = o[t + 1]
    elif pos == 1 and c[t] < entry_px * 0.95:
        pnl.append(o[t + 1] / entry_px - 1 - 0.001)
        pos = 0
print(np.mean(pnl), len(pnl))

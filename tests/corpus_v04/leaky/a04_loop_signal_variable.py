# expect: LA07:high
import numpy as np
import pandas as pd

df = pd.read_csv("es_daily.csv", parse_dates=["date"])
c = df["close"].to_numpy()
o = df["open"].to_numpy()
hi = df["high"].rolling(20).max().shift(1).to_numpy()
pos, entry, pnl = 0, 0.0, []
for t in range(21, len(df) - 1):
    go_long = c[t] > hi[t]
    if go_long and pos == 0:
        pos = 1
        entry = o[t]
    elif pos == 1 and c[t] < entry * 0.95:
        pnl.append(c[t] / entry - 1 - 0.001)
        pos = 0
print(np.mean(pnl), len(pnl))

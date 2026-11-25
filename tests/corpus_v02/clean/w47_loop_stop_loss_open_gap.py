# expect: none
import pandas as pd
df = pd.read_csv("spy.csv")
o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
sma = df["close"].rolling(20).mean().to_numpy()
pos, entry, stop, pnl, fee = 0, 0.0, 0.0, 0.0, 0.0005
for i in range(21, len(df)):
    if pos == 1 and l[i] <= stop:
        exit_px = min(o[i], stop)
        pnl += exit_px / entry - 1 - 2 * fee
        pos = 0
    if pos == 0 and c[i - 1] > sma[i - 1]:
        pos, entry, stop = 1, o[i], o[i] * 0.97
print(pnl)

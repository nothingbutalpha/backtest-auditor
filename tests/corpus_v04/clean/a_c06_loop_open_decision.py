# expect: none
"""Opening-range rule: decide at the open from the gap, fill at that open, exit at the close."""
import pandas as pd

df = pd.read_csv("gld.csv", index_col=0, parse_dates=True)
o = df["open"].to_numpy()
c = df["close"].to_numpy()
trades = []
for i in range(1, len(df)):
    if o[i] > c[i - 1] * 1.01:
        entry_px = o[i]
        trades.append(c[i] / entry_px - 1 - 0.001)
print(sum(trades) / max(len(trades), 1))

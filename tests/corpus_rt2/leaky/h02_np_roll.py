import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
px = df["Close"].to_numpy()
nxt = np.roll(px, -1)
signal = (nxt > px).astype(int)
ret = np.diff(px, prepend=px[0]) / px
pnl = signal * ret
print(pnl.sum())

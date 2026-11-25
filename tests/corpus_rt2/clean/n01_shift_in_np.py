import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
close = df["Close"]
ma = close.rolling(20).mean()
rets = close.pct_change()
pnl = np.where(close.shift(1) > ma.shift(1), rets, 0) - 0.0005
print(pnl.sum())

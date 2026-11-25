# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
c = df["close"].to_numpy()
c_prev = df["close"].shift(1).to_numpy()
signal = (c > c_prev).astype(float)
ret = c / c_prev - 1
pnl = signal * ret - 0.0005
print(np.nanmean(pnl) / np.nanstd(pnl) * np.sqrt(252))

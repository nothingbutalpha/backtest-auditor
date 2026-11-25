# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["signal"] = (df["close"] > df["close"].rolling(50).mean()).astype(float)
pos = df["signal"].shift(1, fill_value=0).values
ret = df["close"].pct_change().fillna(0).values
pnl = pos * ret - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.mean() / pnl.std() * np.sqrt(252))

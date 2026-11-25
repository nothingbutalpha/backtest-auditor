# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np, pandas as pd
c = np.load("close.npy")
sig = pd.Series(c > pd.Series(c).rolling(20).mean().to_numpy())
ret = np.diff(c) / c[:-1]
pos = sig.shift(periods=1).fillna(False).astype(float).to_numpy()
pnl = pos[1:] * ret - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

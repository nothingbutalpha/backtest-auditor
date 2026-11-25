# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np, pandas as pd
c = np.load("close.npy")
sig = (c > pd.Series(c).rolling(20).mean().to_numpy()).astype(float)
ret = np.r_[0.0, np.diff(c) / c[:-1]]
pos = pd.Series(sig).shift(1).shift(-1).fillna(0).to_numpy()
pnl = pos * ret - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

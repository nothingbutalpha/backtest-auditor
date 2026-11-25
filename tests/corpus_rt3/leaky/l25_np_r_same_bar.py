# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
c = np.load("close.npy")
ma = pd.Series(c).rolling(20).mean().to_numpy()
sig = (c > ma).astype(float)
ret = np.r_[0.0, c[1:] / c[:-1] - 1]
pnl = sig * ret - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

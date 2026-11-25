# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np
c = np.load("close.npy")
sig = (c > np.convolve(c, np.ones(20) / 20, "full")[: len(c)]).astype(float)
ret = np.r_[0.0, np.diff(c) / c[:-1]]
pos = np.concatenate([[0.0], sig[2:], [0.0]])
pnl = pos * ret - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
c = np.load("close.npy")
sig = (c > np.convolve(c, np.ones(20) / 20, "full")[: len(c)]).astype(float)
ret = np.r_[0.0, np.diff(c) / c[:-1]]
pos = np.concatenate(([0.0], sig[:-1]))
pnl = pos * ret - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.mean() / pnl.std() * np.sqrt(252))

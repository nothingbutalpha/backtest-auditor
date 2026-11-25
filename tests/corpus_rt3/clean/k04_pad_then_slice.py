# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
c = np.load("close.npy")
sig = (c > np.convolve(c, np.ones(20) / 20, "full")[: len(c)]).astype(float)
ret = np.diff(c) / c[:-1]
pos = np.pad(sig, (1, 0))[:-1]
pnl = pos[1:] * ret - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

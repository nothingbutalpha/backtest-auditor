# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
c = np.load("close.npy")
r = np.diff(c, prepend=c[0]) / np.r_[c[0], c[:-1]]
sig = (c > np.convolve(c, np.ones(10) / 10, "full")[: len(c)]).astype(float)
pos = np.roll(sig, 1); pos[0] = 0
pnl = pos * r - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.mean() / pnl.std() * np.sqrt(252))

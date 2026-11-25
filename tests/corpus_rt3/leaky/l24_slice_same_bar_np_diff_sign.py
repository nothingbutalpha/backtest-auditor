# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
c = np.load("close.npy")
d = np.diff(c)
pnl = np.sign(d) * d / c[:-1] - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

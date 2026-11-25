# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
o = np.load("open.npy"); c = np.load("close.npy")
i, pos, pnl, pending = 1, 0, [], None
while i < len(c):
    if pending is not None:
        pos = pending
        entry = o[i]
        pending = None
    pnl.append(pos * (c[i] / c[i - 1] - 1) - 0.0005 * abs(pos))
    if c[i] > c[i - 1]:
        pending = 1
    i += 1
print(np.mean(pnl) / np.std(pnl) * np.sqrt(252))

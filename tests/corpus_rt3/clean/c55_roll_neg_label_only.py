# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
c = np.load("close.npy")
r = np.diff(c, prepend=c[0]) / c
y = np.roll(r, -1)[:-1]
X = np.column_stack([r, np.roll(r, 1)])[:-1]

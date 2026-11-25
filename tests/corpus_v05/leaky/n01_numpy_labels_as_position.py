# expect: LA13:high
import numpy as np

p = np.load("closes.npy")
H = 10
labels = np.full(len(p), np.nan)
for i in range(len(p) - H):
    labels[i] = 1.0 if p[i + H] / p[i] - 1 > 0.02 else (-1.0 if p[i + H] / p[i] - 1 < -0.02 else 0.0)
valid = ~np.isnan(labels)
print(np.unique(labels[valid], return_counts=True))
ret = np.r_[np.diff(p) / p[:-1], 0.0]
pnl = np.nan_to_num(labels) * ret
print(pnl.sum())

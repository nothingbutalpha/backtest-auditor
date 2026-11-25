# expect: none
import numpy as np

p = np.load("closes.npy")
H = 10
labels = np.full(len(p), np.nan)
for i in range(len(p) - H):
    labels[i] = 1.0 if p[i + H] / p[i] - 1 > 0.02 else (-1.0 if p[i + H] / p[i] - 1 < -0.02 else 0.0)
valid = ~np.isnan(labels)
vals, counts = np.unique(labels[valid], return_counts=True)
print(dict(zip(vals, counts)))
print(np.nanmean(labels), np.count_nonzero(labels == 1.0))
np.save("labels.npy", labels)

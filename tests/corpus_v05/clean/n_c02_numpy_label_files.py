# expect: none
import numpy as np

p = np.load("closes.npy")
H = 10
labels = np.full(len(p), np.nan)
for i in range(len(p) - H):
    labels[i] = 1.0 if p[i + H] / p[i] - 1 > 0.02 else (-1.0 if p[i + H] / p[i] - 1 < -0.02 else 0.0)
np.savez("labels.npz", labels=labels)
labels.tofile("labels.bin")
hist, edges = np.histogram(labels[np.isfinite(labels)], bins=3)
print(hist, np.bincount((labels[np.isfinite(labels)] + 1).astype(int)))

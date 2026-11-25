# expect: LA13:high
import numpy as np
from sklearn.linear_model import LogisticRegression

p = np.load("closes.npy")
H = 10
labels = np.full(len(p), np.nan)
for i in range(len(p) - H):
    labels[i] = 1.0 if p[i + H] / p[i] - 1 > 0.02 else (-1.0 if p[i + H] / p[i] - 1 < -0.02 else 0.0)
print(np.nanmean(labels))
mom = np.r_[np.zeros(20), p[20:] / p[:-20] - 1]
X = np.column_stack([mom, np.nan_to_num(labels)])
y = (np.r_[p[1:] / p[:-1] - 1, 0.0] > 0).astype(int)
LogisticRegression().fit(X[:1000], y[:1000])

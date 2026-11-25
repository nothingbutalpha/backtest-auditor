# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
from sklearn.linear_model import LogisticRegression
close = np.load("close.npy")
ret = np.diff(close) / close[:-1]
feat = np.column_stack([ret[:-1], np.r_[0, ret[:-2]]])
label = (ret[1:] > 0).astype(int)
n = int(len(label) * 0.7)
clf = LogisticRegression().fit(feat[:n], label[:n])
pred = clf.predict(feat[n:])
pnl = np.where(pred == 1, 1, -1) * ret[1:][n:] - 0.0005
print(pnl.mean() / pnl.std() * np.sqrt(252))

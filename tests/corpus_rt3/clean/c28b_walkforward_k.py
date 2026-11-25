# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
from sklearn.linear_model import Ridge
oos = []
for k in range(5):
    X_train, y_train = X[: 500 + 100 * k], y[: 500 + 100 * k]
    X_test, y_test = X[500 + 100 * k: 600 + 100 * k], y[500 + 100 * k: 600 + 100 * k]
    m = Ridge().fit(X_train, y_train)
    oos.append(m.score(X_test, y_test))
print("median", np.median(oos), "range", min(oos), max(oos))

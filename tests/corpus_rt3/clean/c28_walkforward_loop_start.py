# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
from sklearn.linear_model import Ridge
scores = []
for start in range(0, len(X) - 756, 63):
    tr = slice(start, start + 504)
    te = slice(start + 504, start + 567)
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]
    m = Ridge(alpha=1.0).fit(X_train, y_train)
    scores.append(m.score(X_test, y_test))
print("mean OOS R2", np.mean(scores), "best window", max(scores), "worst", min(scores))

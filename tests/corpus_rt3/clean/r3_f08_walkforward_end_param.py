# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import numpy as np
from sklearn.linear_model import Ridge
oos = []
for end in range(504, len(X) - 21, 21):
    X_train, y_train = X[end - 504:end], y[end - 504:end]
    X_test, y_test = X[end:end + 21], y[end:end + 21]
    oos.append(Ridge().fit(X_train, y_train).score(X_test, y_test))
print(np.mean(oos), max(oos))

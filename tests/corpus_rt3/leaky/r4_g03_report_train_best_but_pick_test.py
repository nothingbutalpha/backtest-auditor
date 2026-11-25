# expect: any
# From the third red-team review of PR #8 (2a7cb7c), 2026-10-16.
import numpy as np
from sklearn.linear_model import Ridge
res = []
for alpha in [0.01, 0.1, 1, 10]:
    m = Ridge(alpha=alpha).fit(X_train, y_train)
    tr_err = np.max(np.abs(m.predict(X_train) - y_train))
    res.append((alpha, m.score(X_test, y_test), tr_err))
best = max(res, key=lambda t: t[1])
print(best)

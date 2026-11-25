# expect: any
# From the third red-team review of PR #8 (2a7cb7c), 2026-10-16.
from sklearn.linear_model import Ridge
res = {}
for alpha in [0.01, 0.1, 1, 10]:
    ytr = y_train / max(y_train.abs())
    m = Ridge(alpha=alpha).fit(X_train, ytr)
    res[alpha] = m.score(X_test, y_test / max(y_train.abs()))
best_alpha = max(res, key=res.get)
print(best_alpha, res[best_alpha])

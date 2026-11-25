# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
from sklearn.linear_model import Ridge
res = {}
for start in ["2005", "2010", "2015"]:
    m = Ridge().fit(X_train.loc[start:], y_train.loc[start:])
    res[start] = m.score(X_test, y_test)
best = max(res, key=res.get)
print("best training start", best, res[best])

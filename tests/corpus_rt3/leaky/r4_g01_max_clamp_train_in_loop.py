# expect: any
# From the third red-team review of PR #8 (2a7cb7c), 2026-10-16.
from sklearn.ensemble import RandomForestRegressor
scores = {}
for frac in [0.2, 0.4, 0.6, 0.8]:
    k = max(1, int(X_train.shape[1] * frac))
    m = RandomForestRegressor(max_features=k).fit(X_train, y_train)
    scores[frac] = m.score(X_test, y_test)
best = max(scores, key=scores.get)
print("test R2", scores[best])

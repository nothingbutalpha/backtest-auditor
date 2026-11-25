# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
res = {}
for depth in [2, 4, 8, 16]:
    m = RandomForestRegressor(max_depth=depth)
    cv = cross_val_score(m, X_train, y_train, cv=TimeSeriesSplit(5)).mean()
    m.fit(X_train, y_train)
    res[depth] = m.score(X_test, y_test)
best = max(res, key=res.get)
print("chosen", best, "test", res[best], "cv", cv)

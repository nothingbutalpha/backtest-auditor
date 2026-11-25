# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
tscv = TimeSeriesSplit(n_splits=5)
results = {}
for depth in [3, 5, 8]:
    m = RandomForestRegressor(max_depth=depth, random_state=0)
    cv = cross_val_score(m, X_train, y_train, cv=tscv, scoring="r2").mean()
    m.fit(X_train, y_train)
    results[depth] = (cv, m.score(X_test, y_test))
best_depth = max(results, key=lambda d: results[d][0])
print(best_depth, results[best_depth])

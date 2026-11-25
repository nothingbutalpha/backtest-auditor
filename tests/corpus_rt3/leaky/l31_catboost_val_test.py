# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from catboost import CatBoostRegressor
m = CatBoostRegressor(iterations=3000)
m.fit(X_train, y_train, eval_set=(X_test, y_test))

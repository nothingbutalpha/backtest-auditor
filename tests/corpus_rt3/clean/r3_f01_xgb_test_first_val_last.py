# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import xgboost as xgb
m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=50)
m.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_val, y_val)], verbose=False)
print(m.score(X_test, y_test))

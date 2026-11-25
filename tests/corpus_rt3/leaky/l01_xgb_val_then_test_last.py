# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import xgboost as xgb
m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=50)
m.fit(X_train, y_train, eval_set=[(X_val, y_val), (X_test, y_test)])
print(m.score(X_test, y_test))

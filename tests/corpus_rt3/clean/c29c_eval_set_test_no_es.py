# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import xgboost as xgb
m = xgb.XGBRegressor(n_estimators=300)
m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
curves = m.evals_result()

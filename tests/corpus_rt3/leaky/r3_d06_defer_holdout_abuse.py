# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import xgboost as xgb
m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=50)
m.fit(X_train, y_train, eval_set=[(X_oos, y_oos)])
print("OOS R2", m.score(X_oos, y_oos))
live_pred = m.predict(X_test_live)

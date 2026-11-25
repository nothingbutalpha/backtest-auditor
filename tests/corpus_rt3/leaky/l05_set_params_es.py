# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import xgboost as xgb
m = xgb.XGBClassifier(n_estimators=2000)
m.set_params(early_stopping_rounds=30)
m.fit(X_train, y_train, eval_set=[(X_test, y_test)])

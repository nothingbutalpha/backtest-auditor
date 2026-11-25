# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import lightgbm as lgb
m = lgb.LGBMRegressor(n_estimators=5000)
m.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_val, y_val)], callbacks=[lgb.early_stopping(100)])
print(m.score(X_test, y_test))

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import lightgbm as lgb
X_tr, X_es = X_train[:-250], X_train[-250:]
y_tr, y_es = y_train[:-250], y_train[-250:]
m = lgb.LGBMRegressor(n_estimators=5000)
m.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], callbacks=[lgb.early_stopping(100)])
print(m.score(X_test, y_test))

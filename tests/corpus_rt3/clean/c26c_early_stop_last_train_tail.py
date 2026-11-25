# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import xgboost as xgb
split = int(len(X_train) * 0.85)
m = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=30)
m.fit(X_train[:split], y_train[:split], eval_set=[(X_train[split:], y_train[split:])])
proba_test = m.predict_proba(X_test)[:, 1]

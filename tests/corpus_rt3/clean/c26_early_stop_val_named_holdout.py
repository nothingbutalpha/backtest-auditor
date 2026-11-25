# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import xgboost as xgb
X_train, y_train = X.loc[:"2016"], y.loc[:"2016"]
X_holdout, y_holdout = X.loc["2017"], y.loc["2017"]
X_test, y_test = X.loc["2018":], y.loc["2018":]
model = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_holdout, y_holdout)])
print(model.score(X_test, y_test))

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test)
bst = xgb.train(params, dtrain, 2000, evals=[(dtrain, "train"), (dvalid, "eval")], early_stopping_rounds=50)
pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

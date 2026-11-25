# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import lightgbm as lgb
dtrain = lgb.Dataset(X_tr, y_tr)
dval = lgb.Dataset(X_va, y_va, reference=dtrain)
booster = lgb.train(params, dtrain, valid_sets=[dtrain, dval], valid_names=["train", "valid"],
                    callbacks=[lgb.early_stopping(100)])
pred_test = booster.predict(X_test, num_iteration=booster.best_iteration)

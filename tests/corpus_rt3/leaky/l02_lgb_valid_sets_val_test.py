# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import lightgbm as lgb
booster = lgb.train(params, lgb.Dataset(X_tr, y_tr), valid_sets=[lgb.Dataset(X_va, y_va), lgb.Dataset(X_te, y_te)],
                    callbacks=[lgb.early_stopping(100)])

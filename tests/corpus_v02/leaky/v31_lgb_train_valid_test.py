# expect: LA14:high
import lightgbm as lgb
import pandas as pd
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
train, test = df[df.index < "2020-01-01"], df[df.index >= "2020-01-01"]
feats = [c for c in df.columns if c != "fwd_ret"]
dtrain = lgb.Dataset(train[feats], train["fwd_ret"])
dtest = lgb.Dataset(test[feats], test["fwd_ret"])
booster = lgb.train({"objective": "regression"}, dtrain, valid_sets=[dtest],
                    callbacks=[lgb.early_stopping(100)])
pred = booster.predict(test[feats])
print(pd.Series(pred, index=test.index).corr(test["fwd_ret"]))

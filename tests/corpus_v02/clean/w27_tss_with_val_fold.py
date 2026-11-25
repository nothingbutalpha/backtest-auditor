# expect: none
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
feats = [c for c in df.columns if c != "fwd_ret"]
holdout = df.loc["2021":]
dev = df.loc[:"2020"]
for tr_idx, va_idx in TimeSeriesSplit(n_splits=5, gap=5).split(dev):
    tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]
    booster = lgb.train({"objective": "regression"}, lgb.Dataset(tr[feats], tr["fwd_ret"]),
                        valid_sets=[lgb.Dataset(va[feats], va["fwd_ret"])], callbacks=[lgb.early_stopping(50)])
print(booster.predict(holdout[feats])[:5])

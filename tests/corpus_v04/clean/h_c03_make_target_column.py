# expect: none
import numpy as np
import pandas as pd
import lightgbm as lgb


def make_target(close, h=5):
    out = []
    for i in range(len(close)):
        if i + h < len(close) and close[i + h] > close[i]:
            out.append(1)
        else:
            out.append(0)
    return out


df = pd.read_parquet("aapl.parquet")
df["rsi_lag"] = df["close"].diff().clip(lower=0).rolling(14).mean().shift(1)
df["target"] = make_target(df["close"].to_numpy())
train = df.iloc[:3000].dropna()
booster = lgb.train({"objective": "binary"}, lgb.Dataset(train[["rsi_lag"]], label=train["target"]))
print(booster.num_trees())

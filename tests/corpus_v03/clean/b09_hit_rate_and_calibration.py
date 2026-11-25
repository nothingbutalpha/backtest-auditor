# expect: none
# Class 2: forward returns used to grade out-of-sample predictions.
import numpy as np
import pandas as pd

df = pd.read_parquet("preds.parquet")
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["up"] = (df["fwd_ret"] > 0).astype(int)
oos = df[df["date"] >= "2022-01-01"].dropna()
bins = pd.cut(oos["proba"], [0, 0.4, 0.5, 0.6, 1.0])
calib = oos.groupby(bins)["up"].mean()
hit = (np.sign(oos["proba"] - 0.5) == np.sign(oos["fwd_ret"])).mean()
print(calib, hit)

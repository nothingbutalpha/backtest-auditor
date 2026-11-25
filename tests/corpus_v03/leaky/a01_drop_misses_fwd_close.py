# expect: LA10:high
# Twin of clean/a01: the drop list forgets fwd_close, so tomorrow's close is a feature.
import pandas as pd
from lightgbm import LGBMClassifier

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
g = df.groupby("ticker")["close"]
df["fwd_close"] = g.shift(-1)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > 0).astype(int)
df["mom_20"] = g.pct_change(20)
df = df.dropna()
X = df.drop(columns=["fwd_ret", "label", "date", "ticker"])
y = df["label"]
train = df["date"] < "2019-01-01"
model = LGBMClassifier(n_estimators=200).fit(X[train], y[train])

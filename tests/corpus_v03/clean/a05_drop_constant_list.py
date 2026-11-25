# expect: none
# Class 1: the columns to drop live in a module-level constant.
import pandas as pd
from xgboost import XGBClassifier

LEAK_COLS = ["fwd_close", "fwd_ret", "label"]
ID_COLS = ["date", "ticker"]

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
grp = df.groupby("ticker")["close"]
df["fwd_close"] = grp.shift(-21)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > df.groupby("date")["fwd_ret"].transform("median")).astype(int)
df["mom_12_1"] = grp.shift(21) / grp.shift(252) - 1
df["rev_1m"] = grp.pct_change(21)
df = df.dropna()
X = df.drop(columns=LEAK_COLS + ID_COLS)
y = df["label"]
tr = df["date"] < "2017-01-01"
m = XGBClassifier(n_estimators=400, max_depth=3).fit(X[tr], y[tr])

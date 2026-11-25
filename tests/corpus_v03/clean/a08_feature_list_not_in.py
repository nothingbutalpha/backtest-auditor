# expect: none
# Class 1: feature list = all columns not in an explicit exclusion tuple.
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("d.csv", parse_dates=["date"])
df["fwd_close"] = df.groupby("ticker")["close"].shift(-1)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > 0).astype(int)
df["vol"] = df.groupby("ticker")["close"].pct_change().rolling(20).std()
EXCLUDE = ("fwd_close", "fwd_ret", "label", "date", "ticker")
feats = [c for c in df.columns if c not in EXCLUDE]
tr = df["date"] < "2019-06-30"
gb = GradientBoostingClassifier().fit(df.loc[tr, feats], df.loc[tr, "label"])

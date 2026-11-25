# expect: LA10:high
# Twin of clean/a08: the exclusion tuple only lists the label.
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("d.csv", parse_dates=["date"])
df["fwd_close"] = df.groupby("ticker")["close"].shift(-1)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > 0).astype(int)
df["vol"] = df.groupby("ticker")["close"].pct_change().rolling(20).std()
feats = [c for c in df.columns if c not in ("label", "date", "ticker")]
tr = df["date"] < "2019-06-30"
gb = GradientBoostingClassifier().fit(df.loc[tr, feats], df.loc[tr, "label"])

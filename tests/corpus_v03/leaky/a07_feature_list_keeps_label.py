# expect: LA10:high
# Twin of clean/a07: prefix filter only; the label column stays in the features.
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("panel.parquet")
c = df.groupby("ticker")["close"]
df["fwd_close"] = c.shift(-1)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > 0).astype(int)
df["mom"] = c.pct_change(20)
df = df.dropna()
features = [col for col in df.columns if not col.startswith("fwd_") and col not in ("date", "ticker")]
train = df[df["date"] < "2020-01-01"]
lr = LogisticRegression().fit(train[features], train["label"])

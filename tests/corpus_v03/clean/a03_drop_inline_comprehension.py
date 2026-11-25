# expect: none
# Class 1: drop list built inline, prefix filter plus the label.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("f.parquet")
df["fwd_ret_1"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
df["fwd_ret_5"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["label"] = (df["fwd_ret_5"] > 0).astype(int)
df["ma_gap"] = df["close"] / df.groupby("ticker")["close"].transform(lambda s: s.rolling(50).mean()) - 1
df = df.dropna()
X = df.drop(columns=[c for c in df.columns if c.startswith("fwd_") or c in ("label", "date", "ticker")])
y = df["label"]
is_train = df["date"] < "2018-12-31"
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50).fit(X[is_train], y[is_train])

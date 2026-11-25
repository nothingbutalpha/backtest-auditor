# expect: none
# From the red-team review of v0.1 (regression case).
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
df = pd.read_parquet("d.parquet")
df["y"] = (df["close"].shift(-5) > df["close"]).astype(int)
feats = ["f1", "f2", "f3"]
train = df[df.index < "2020-01-01"].dropna(); test = df[df.index >= "2020-01-01"].dropna()
scaler = StandardScaler().fit(train[feats])
clf = GradientBoostingClassifier().fit(scaler.transform(train[feats]), train["y"])
test_pred = clf.predict(scaler.transform(test[feats]))

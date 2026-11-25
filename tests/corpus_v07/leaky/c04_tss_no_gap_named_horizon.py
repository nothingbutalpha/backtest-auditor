# expect: LA06@13
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

STEP = 10
df = pd.read_parquet("eurusd_daily.parquet")
df["up"] = (df["close"].shift(-STEP) > df["close"]).astype(int)
df = df.iloc[:-STEP]
X = df[["carry", "mom_20", "rr_25d"]].values
y = df["up"].values
acc = []
for tr, te in TimeSeriesSplit(n_splits=6).split(X):
    m = LogisticRegression().fit(X[tr], y[tr])
    acc.append(m.score(X[te], y[te]))
print(acc)

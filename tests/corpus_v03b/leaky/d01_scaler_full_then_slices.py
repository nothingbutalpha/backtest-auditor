# expect: LA05
# Twin of clean/d01: the scaler is fitted on every row, the model on the slice, the score on the later slice.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("spy_features.csv", index_col="date", parse_dates=True)
X = df[["mom_20", "vol_20", "rsi_14"]]
y = (df["close"].pct_change().shift(-1) > 0).astype(int)
cut = "2019-01-14"
scaler = StandardScaler().fit(X)
model = LogisticRegression().fit(scaler.transform(X.loc[:cut]), y.loc[:cut])
print(model.score(scaler.transform(X.loc["2019-01-15":]), y.loc["2019-01-15":]))

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["target"] = (df["ret"].shift(-1) > 0).astype(int)
X = df[["ret"]].fillna(0).values
y = df["target"].values
model = RandomForestClassifier()
model.fit(X, y)
tscv = TimeSeriesSplit(n_splits=5)
for tr, te in tscv.split(X):
    preds = model.predict(X[te])
    print((preds == y[te]).mean())

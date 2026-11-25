import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["lag1"] = df["ret"].shift(1)
df["target"] = (df["ret"].shift(-1) > 0).astype(int)
df = df.dropna()
X, y = df[["ret", "lag1"]].values, df["target"].values
for tr, te in TimeSeriesSplit(n_splits=5).split(X):
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X[tr], y[tr])
    print(model.score(X[te], y[te]))

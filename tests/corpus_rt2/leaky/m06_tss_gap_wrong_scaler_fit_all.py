import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
feats = pd.DataFrame({"r": df["Close"].pct_change(), "v": df["Volume"].pct_change()}).dropna()
y = (df["Close"].pct_change().shift(-1).reindex(feats.index) > 0).astype(int).values
sc = StandardScaler()
Z = sc.fit_transform(feats)
for tr, te in TimeSeriesSplit(n_splits=4).split(Z):
    m = SVC().fit(Z[tr], y[tr])
    print(m.score(Z[te], y[te]))

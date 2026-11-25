# expect: none
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
r = pd.read_csv("spy.csv", index_col=0)["close"].pct_change().dropna()
feats = pd.concat([r.rolling(21).std(), r.rolling(63).mean()], axis=1).dropna()
km = KMeans(n_clusters=3, n_init=10).fit(feats.loc[:"2014"])
labels = km.predict(feats)
calm = pd.Series(labels == np.argmin(km.cluster_centers_[:, 0]), index=feats.index)
exposure = calm.astype(float).shift(1).loc["2015":]
pnl = exposure * r.reindex(exposure.index)
print(pnl.sum())

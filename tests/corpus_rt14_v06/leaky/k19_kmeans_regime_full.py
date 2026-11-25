# leak: 7
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
r = pd.read_csv("spy.csv", index_col=0)["close"].pct_change().dropna()
feats = pd.concat([r.rolling(21).std(), r.rolling(63).mean()], axis=1).dropna()
labels = KMeans(n_clusters=3, n_init=10).fit_predict(feats)
calm = pd.Series(labels == np.argmin([feats.iloc[:, 0][labels == k].mean() for k in range(3)]), index=feats.index)
exposure = calm.astype(float).shift(1)
pnl = exposure * r.reindex(feats.index)
print(pnl.sum())

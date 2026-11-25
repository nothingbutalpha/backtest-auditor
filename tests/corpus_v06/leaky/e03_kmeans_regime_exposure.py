# expect: LA05:medium@9
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

px = pd.read_csv("qqq.csv", index_col=0, parse_dates=True)["close"]
r = px.pct_change()
f = pd.DataFrame({"vol": r.rolling(21).std(), "trend": px.pct_change(63)}).dropna()
f["regime"] = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(f[["vol", "trend"]])
calm = f.groupby("regime")["vol"].mean().idxmin()
exposure = (f["regime"] == calm).astype(float).shift(1)
print((exposure * r.reindex(f.index)).mean() * 252)

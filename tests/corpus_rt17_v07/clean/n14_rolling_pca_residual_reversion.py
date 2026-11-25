# expect: none
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
r = pd.read_parquet("sector_etf_rets.parquet")
pnl = []
for t in range(252, len(r) - 1):
    past = r.iloc[t - 252:t]
    pca = PCA(n_components=3).fit(past)
    today = r.iloc[[t]]
    resid = today - pca.inverse_transform(pca.transform(today))
    w = -np.sign(resid.iloc[0]) / r.shape[1]
    pnl.append(float(r.iloc[t + 1] @ w))
print(sum(pnl))

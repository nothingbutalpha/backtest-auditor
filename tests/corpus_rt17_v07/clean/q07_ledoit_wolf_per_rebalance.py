# expect: none
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
rets = pd.read_parquet("sector_etf_rets.parquet")
out = []
for t in range(250, len(rets), 21):
    cov = LedoitWolf().fit(rets.iloc[t - 250:t]).covariance_
    inv = np.linalg.pinv(cov)
    w = inv.sum(axis=1) / inv.sum()
    out.append(rets.iloc[t:t + 21] @ w)
print(pd.concat(out).add(1).prod())

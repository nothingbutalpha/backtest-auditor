# expect: LA05:medium@7
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

R = pd.read_parquet("reit_returns.parquet").fillna(0.0)
pca = PCA(n_components=4).fit(R.to_numpy())
resid = pd.DataFrame(R.to_numpy() - pca.inverse_transform(pca.transform(R.to_numpy())), index=R.index, columns=R.columns)
s = resid.rolling(15).sum()
w = -s.sub(s.mean(axis=1), axis=0)
w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
pnl = (w * R).sum(axis=1)
print(pnl.mean() / pnl.std() * np.sqrt(252))

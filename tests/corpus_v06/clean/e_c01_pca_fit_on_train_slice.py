# expect: none
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

R = pd.read_parquet("reit_returns.parquet").fillna(0.0)
train = R.loc[:"2015"]
pca = PCA(n_components=4).fit(train.to_numpy())
test = R.loc["2016":]
resid = pd.DataFrame(test.to_numpy() - pca.inverse_transform(pca.transform(test.to_numpy())), index=test.index,
                     columns=test.columns)
s = resid.rolling(15).sum()
w = -s.sub(s.mean(axis=1), axis=0)
w = w.div(w.abs().sum(axis=1), axis=0).shift(1)
pnl = (w * test).sum(axis=1)
print(pnl.mean() / pnl.std() * np.sqrt(252))

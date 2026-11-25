# expect: LA05:high
# Twin of clean/d08: PCA fitted on every row, its output sliced into fit and evaluation rows.
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

X = pd.read_parquet("returns_wide.parquet").fillna(0)
y = pd.read_parquet("target.parquet")["fwd_ret"]
n_fit = 750
Z = PCA(n_components=5).fit_transform(X)
lr = LinearRegression().fit(Z[:n_fit], y.iloc[:n_fit])
print(lr.score(Z[n_fit:], y.iloc[n_fit:]))

# expect: none
# LA05 class: fit_transform on the first n rows, transform on the rest.
import pandas as pd
from sklearn.decomposition import PCA

X = pd.read_parquet("returns_wide.parquet").fillna(0)
n_fit = 750
pca = PCA(n_components=5)
Z_fit = pca.fit_transform(X.iloc[:n_fit])
Z_new = pca.transform(X.iloc[n_fit:])

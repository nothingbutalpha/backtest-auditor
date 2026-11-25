# expect: none
"""How many factors explain sector returns? A descriptive report, no trading."""
import pandas as pd
from sklearn.decomposition import PCA

R = pd.read_parquet("sector_returns.parquet").dropna()
pca = PCA(n_components=5).fit(R.to_numpy())
print(pd.Series(pca.explained_variance_ratio_, index=[f"pc{i + 1}" for i in range(5)]).round(3))

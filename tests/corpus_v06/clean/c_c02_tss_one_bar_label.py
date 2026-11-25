# expect: none
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_parquet("bund_features.parquet")
df["y"] = df["close"].pct_change().shift(-1)
df = df.dropna()
X = df.drop(columns=["y", "close"])
scores = [Ridge().fit(X.iloc[tr], df["y"].iloc[tr]).score(X.iloc[te], df["y"].iloc[te])
          for tr, te in TimeSeriesSplit(n_splits=6).split(X)]
print(np.mean(scores))

# expect: none
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

H = 15
df = pd.read_parquet("bund_features.parquet")
df["y"] = df["close"].shift(-H) / df["close"] - 1
df = df.dropna()
X = df.drop(columns=["y", "close"])
scores = []
for tr, te in TimeSeriesSplit(n_splits=6, gap=H).split(X):
    rf = RandomForestRegressor(n_estimators=200).fit(X.iloc[tr], df["y"].iloc[tr])
    scores.append(rf.score(X.iloc[te], df["y"].iloc[te]))
print(np.mean(scores))

# expect: none
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
d = pd.read_parquet("factors.parquet")
y = d.pop("ret_next")
scores = []
for a, b in TimeSeriesSplit(5, gap=1).split(d):
    mi = pd.Series(mutual_info_regression(d.iloc[a], y.iloc[a]), index=d.columns)
    cols = mi.nlargest(5).index
    lr = LinearRegression().fit(d.iloc[a][cols], y.iloc[a])
    scores.append(lr.score(d.iloc[b][cols], y.iloc[b]))
print(np.mean(scores))

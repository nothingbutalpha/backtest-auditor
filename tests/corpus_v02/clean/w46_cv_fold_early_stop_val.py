# expect: none
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
X, y = df.drop(columns="fwd_ret").to_numpy(), df["fwd_ret"].to_numpy()
scores = []
for tr_idx, te_idx in TimeSeriesSplit(n_splits=5).split(X):
    cut = int(len(tr_idx) * 0.8)
    fit_idx, val_idx = tr_idx[:cut], tr_idx[cut:]
    m = XGBRegressor(n_estimators=1000, early_stopping_rounds=30)
    m.fit(X[fit_idx], y[fit_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
    scores.append(np.corrcoef(m.predict(X[te_idx]), y[te_idx])[0, 1])
print(np.mean(scores))

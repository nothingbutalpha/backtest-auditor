# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
from sklearn.model_selection import ParameterGrid
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
res = []
for p in ParameterGrid({"fast": [5, 10, 20], "slow": [50, 100, 200]}):
    sig = (px.rolling(p["fast"]).mean() > px.rolling(p["slow"]).mean()).shift(1)
    r = sig * px.pct_change() - 0.0005 * sig.diff().abs()
    res.append((r.mean() / r.std() * np.sqrt(252), p))
print(max(res, key=lambda t: t[0]))

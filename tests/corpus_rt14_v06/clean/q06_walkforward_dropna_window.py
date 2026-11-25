# expect: none
import numpy as np
import pandas as pd
px = pd.read_parquet("crsp_wide.parquet")
dates = px.index[252::21]
out = []
for d in dates:
    window = px.loc[:d].iloc[-252:].dropna(axis=1)
    vol = window.pct_change().std()
    w = (1 / vol) / (1 / vol).sum()
    nxt = px.loc[d:].iloc[1:22][w.index].pct_change().sum()
    out.append((w * nxt).sum())
print(np.mean(out))

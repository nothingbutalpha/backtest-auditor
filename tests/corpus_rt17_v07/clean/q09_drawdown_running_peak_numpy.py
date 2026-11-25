# expect: none
import numpy as np
import pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
peak = np.maximum.accumulate(px.values)
dd = pd.Series(px.values / peak - 1, index=px.index)
pos = (dd > -0.1).astype(float).shift(1)
print((pos * px.pct_change()).sum())

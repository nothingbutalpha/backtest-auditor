# expect: none
import numpy as np
import pandas as pd
def zscore(s, n=60):
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - m) / sd
px = pd.read_csv("tlt.csv", index_col=0, parse_dates=True)["close"]
z = zscore(px)
pos = -np.sign(z).shift(1)
pnl = pos * px.pct_change()
print(pnl.sum())

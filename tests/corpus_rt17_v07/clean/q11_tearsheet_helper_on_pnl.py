# expect: none
import numpy as np
import pandas as pd
def tearsheet(r):
    mu, sd = r.mean(), r.std()
    return {"ann_ret": mu * 252, "ann_vol": sd * np.sqrt(252), "worst_z": ((r - mu) / sd).min()}
px = pd.read_csv("iwm.csv", index_col=0, parse_dates=True)["close"]
pos = (px > px.rolling(200).mean()).astype(float).shift(1)
pnl = pos * px.pct_change()
print(tearsheet(pnl.dropna()))

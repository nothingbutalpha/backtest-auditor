# expect: none
import numpy as np
import pandas as pd
px = pd.read_csv("agg.csv", index_col=0, parse_dates=True)["close"]
r = px.pct_change()
vol = r.rolling(60).std() * np.sqrt(252)
lev = (0.10 / vol).clip(upper=2.0)
lev_m = lev.resample("M").last().reindex(r.index, method="ffill").shift(1)
print((lev_m * r).add(1).prod())

# expect: none
# CO01 class: Sharpe of a volatility-target overlay on an index series.
import numpy as np
import pandas as pd

spx = pd.read_csv("spx.csv", index_col="date", parse_dates=True)["close"]
ret = spx.pct_change()
vol = ret.rolling(20).std() * np.sqrt(252)
lev = (0.10 / vol).clip(upper=2.0).shift(1)
overlay = lev * ret
print("overlay Sharpe", overlay.mean() / overlay.std() * np.sqrt(252))

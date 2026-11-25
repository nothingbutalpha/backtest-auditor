# expect: none
import pandas as pd
from scipy.stats import percentileofscore
vix = pd.read_csv("vix.csv", index_col=0, parse_dates=True)["close"]
pct = vix.rolling(252).apply(lambda w: percentileofscore(w[:-1], w[-1]), raw=True)
spy = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
pos = (pct > 80).astype(float).shift(1)
print((pos * spy.pct_change()).sum())

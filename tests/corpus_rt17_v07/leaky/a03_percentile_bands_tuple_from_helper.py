# leak: 5
import numpy as np
import pandas as pd
def entry_bands(x):
    lo, hi = np.nanpercentile(x, [10, 90])
    return lo, hi
a = pd.read_csv("ko.csv", index_col=0, parse_dates=True)["close"]
b = pd.read_csv("pep.csv", index_col=0, parse_dates=True)["close"]
spread = np.log(a) - np.log(b)
lower, upper = entry_bands(spread.values)
side = pd.Series(np.where(spread < lower, 1, np.where(spread > upper, -1, 0)), index=spread.index)
pnl = side.shift(1) * spread.diff()
print(pnl.cumsum().iloc[-1])

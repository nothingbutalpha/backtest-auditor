# expect: LA05|LA09@7
import numpy as np
import pandas as pd

def trend_gap(px):
    x = np.arange(len(px))
    slope, icpt = np.polyfit(x, np.log(px.values), 1)
    return pd.Series(np.log(px.values) - (slope * x + icpt), index=px.index)

copper = pd.read_csv("hg.csv", index_col=0, parse_dates=True)["close"]
gap = trend_gap(copper)
signal = -np.sign(gap)
pnl = signal.shift(1) * copper.pct_change()
print(pnl.mean() / pnl.std() * 252 ** 0.5)

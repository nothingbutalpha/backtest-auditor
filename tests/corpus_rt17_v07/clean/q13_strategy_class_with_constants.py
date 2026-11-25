# expect: none
import numpy as np
import pandas as pd
class Breakout:
    def __init__(self, lookback):
        self.lookback = lookback
        self.ann = np.sqrt(252)
    def signal(self, px):
        hi = px.rolling(self.lookback).max().shift(1)
        return (px > hi).astype(float)
px = pd.read_csv("hg.csv", index_col=0, parse_dates=True)["close"]
s = Breakout(55)
pnl = s.signal(px).shift(1) * px.pct_change()
print(pnl.mean() / pnl.std() * s.ann)

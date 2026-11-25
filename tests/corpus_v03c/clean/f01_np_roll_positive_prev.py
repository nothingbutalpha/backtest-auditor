# expect: none
# Twin of leaky/f01: the previous close, first element masked.
import numpy as np
import pandas as pd

close = pd.read_csv("spy.csv")["close"].to_numpy()
ret = np.zeros_like(close)
ret[1:] = close[1:] / close[:-1] - 1
prev = np.roll(close, 1)
prev[0] = np.nan
up = close > prev
pos = np.zeros_like(close)
pos[1:] = np.where(up[:-1], 1, -1)
pnl = pos * ret - 0.0005

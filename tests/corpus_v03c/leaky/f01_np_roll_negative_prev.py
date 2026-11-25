# expect: LA01|LA10:high
# np.roll(close, -1) is the NEXT close, misnamed `prev`.
import numpy as np
import pandas as pd

close = pd.read_csv("spy.csv")["close"].to_numpy()
ret = np.zeros_like(close)
ret[1:] = close[1:] / close[:-1] - 1
prev = np.roll(close, -1)
up = close > prev
pos = np.zeros_like(close)
pos[1:] = np.where(up[:-1], 1, -1)
pnl = pos * ret - 0.0005

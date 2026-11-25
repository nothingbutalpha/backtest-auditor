# expect: LA02:high@7
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

px = pd.read_csv("ko_daily.csv", index_col=0, parse_dates=True)["close"]
lows = argrelextrema(px.to_numpy(), np.less, order=10)[0]
signal = pd.Series(0.0, index=px.index)
signal.iloc[lows] = 1.0
position = signal.rolling(5, min_periods=1).max().shift(1)
print((position * px.pct_change()).sum())

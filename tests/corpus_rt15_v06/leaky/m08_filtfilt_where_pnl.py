# leak: 7
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
px = pd.read_csv("zn.csv")["close"]
b, a = butter(2, 0.1)
trend = filtfilt(b, a, px.to_numpy())
up = trend > np.roll(trend, 1)
ret = px.pct_change().to_numpy()
pnl = np.where(up, ret, -ret)
print(np.nansum(pnl))

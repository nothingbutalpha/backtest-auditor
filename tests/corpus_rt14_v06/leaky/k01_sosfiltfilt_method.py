# leak: 7
import numpy as np
import pandas as pd
from scipy import signal as sg
px = pd.read_csv("es.csv", index_col=0, parse_dates=True)["close"]
sos = sg.butter(3, 0.05, output="sos")
smooth = pd.Series(sg.sosfiltfilt(sos, px.values), index=px.index)
position = np.sign(smooth.diff())
strategy_ret = position.shift(1) * px.pct_change()
print(strategy_ret.sum())

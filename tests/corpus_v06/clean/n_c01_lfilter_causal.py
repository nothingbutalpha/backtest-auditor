# expect: none
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

px = pd.read_csv("ng_daily.csv", index_col=0, parse_dates=True)["settle"]
b, a = butter(2, 0.08)
smooth = pd.Series(lfilter(b, a, px.to_numpy()), index=px.index)
position = np.sign(smooth.diff()).shift(1)
pnl = position * px.pct_change() - 0.0003 * position.diff().abs()
print(pnl.mean() / pnl.std() * np.sqrt(252))

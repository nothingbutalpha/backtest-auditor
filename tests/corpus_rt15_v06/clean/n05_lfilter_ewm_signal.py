# expect: none
import numpy as np
import pandas as pd
from scipy.signal import lfilter
px = pd.read_csv("zn.csv")["close"]
alpha = 0.1
trend = lfilter([alpha], [1, alpha - 1], px.to_numpy())
up = pd.Series(trend).diff() > 0
pnl = up.shift(1).fillna(False).astype(int) * px.pct_change()
print(pnl.sum())

# expect: none
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
df = pd.read_csv("btc_1h.csv")
order = 5
lows = argrelextrema(df["close"].to_numpy(), np.less_equal, order=order)[0]
df["low_mark"] = 0
df.loc[lows, "low_mark"] = 1
df["signal"] = df["low_mark"].shift(order)
df["pnl"] = df["signal"].shift(1) * df["close"].pct_change()
print(df["pnl"].sum())

# leak: 6
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
df = pd.read_csv("eurusd_1h.csv")
idx = argrelextrema(df["close"].to_numpy(), np.greater, order=8)[0]
df["top"] = 0
df.loc[idx, "top"] = 1
df["short"] = -df["top"].shift(3)
df["pnl"] = df["short"].shift(1) * df["close"].pct_change()
print(df["pnl"].sum())

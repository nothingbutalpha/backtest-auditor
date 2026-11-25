# leak: 7
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
df = pd.read_csv("btc_1h.csv")
c = df["close"].to_numpy()
peaks, _ = find_peaks(c, distance=12)
df["signal"] = 0
df.loc[peaks, "signal"] = -1
df["pnl"] = df["signal"].shift(1) * df["close"].pct_change()
print(df["pnl"].cumsum().iloc[-1])

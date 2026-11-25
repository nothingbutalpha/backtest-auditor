# expect: none
import pandas as pd
from scipy.signal import savgol_filter
def last_smoothed(w):
    return savgol_filter(w, len(w), 3)[-1]
px = pd.read_csv("btc.csv", index_col=0)["close"]
smooth = px.rolling(41).apply(last_smoothed, raw=True)
signal = (px > smooth).astype(int)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())

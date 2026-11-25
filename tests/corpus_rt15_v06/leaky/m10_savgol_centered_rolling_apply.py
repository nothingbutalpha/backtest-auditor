# leak: 5
import pandas as pd
from scipy.signal import savgol_filter
px = pd.read_csv("btc.csv", index_col=0)["close"]
smooth = px.rolling(21, center=True).apply(lambda w: savgol_filter(w, 21, 2)[-1], raw=True)
signal = (px > smooth).astype(int)
pnl = signal.shift(1) * px.pct_change()
print(pnl.sum())

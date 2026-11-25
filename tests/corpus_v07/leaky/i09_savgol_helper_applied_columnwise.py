# expect: LA02@6
import pandas as pd
from scipy.signal import savgol_filter

def slope_of(s):
    d = savgol_filter(s.values, 21, 2, deriv=1)
    return pd.Series(d, index=s.index)

px = pd.read_parquet("crypto_4h.parquet")
slopes = px.apply(slope_of)
w = slopes.div(slopes.abs().sum(axis=1), axis=0)
print((w.shift(1) * px.pct_change()).sum(axis=1).sum())

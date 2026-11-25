# expect: none
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth(s):
    return pd.Series(savgol_filter(s.values, 21, 2), index=s.index)

px = pd.read_parquet("crypto_4h.parquet")["BTC"]
ax = px.plot(alpha=0.4)
smooth(px).plot(ax=ax)
plt.savefig("btc_smoothed.png")

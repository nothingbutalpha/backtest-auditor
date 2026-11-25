# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", parse_dates=["date"]).sort_values("date")
c = df["close"].to_numpy()
ma = pd.Series(c).rolling(50).mean().to_numpy()
signal = (c > ma).astype(float)
ret = np.r_[np.nan, c[1:] / c[:-1] - 1]
pos = np.r_[0.0, signal[:-1]]
strat = pos * ret
cost = 0.0005 * np.abs(np.diff(pos, prepend=0))
net = strat - cost
print(np.nanmean(net) / np.nanstd(net) * np.sqrt(252))

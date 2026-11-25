# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
close = np.loadtxt("close.txt")
mom = np.zeros_like(close)
mom[20:] = close[20:] / close[:-20] - 1
signal = np.sign(mom)
lagged = np.insert(signal[:-1], 0, 0.0)
ret = np.append(0.0, np.diff(np.log(close)))
pnl = lagged * ret - 0.0003 * np.abs(np.diff(lagged, prepend=0.0))
print("sharpe", pnl.mean() / pnl.std() * 252 ** 0.5)

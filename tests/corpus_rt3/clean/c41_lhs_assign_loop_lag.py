# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
close = np.load("close.npy")
sma = np.convolve(close, np.ones(20) / 20, mode="full")[: len(close)]
pos = np.zeros(len(close))
for t in range(20, len(close) - 1):
    pos[t + 1] = 1.0 if close[t] > sma[t] else 0.0
ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1
pnl = pos * ret - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.mean() / pnl.std() * np.sqrt(252))

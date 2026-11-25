# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
# v0.6: the moving average was np.convolve(..., "same"), a centred window (4 bars ahead), so this "correct"
# file had a real leak; now a trailing average, as in the other corpus files. The centred version is v06/n07.
import numpy as np
close = np.load("close.npy")
ret = np.diff(close) / close[:-1]
z = close - np.convolve(close, np.ones(10) / 10, "full")[: len(close)]
signal = (z < 0).astype(float)
pos = np.pad(signal, (1, 0))[:-1]
strat = pos[1:] * ret - 0.0005 * np.abs(np.diff(pos))
print(strat.mean() / strat.std() * np.sqrt(252))

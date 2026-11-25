# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
close = np.load("close.npy")
sma = np.convolve(close, np.ones(20) / 20, mode="full")[: len(close)]
today = close[:-1]
tomorrow = close[1:]
signal = today > sma[:-1]
fwd = tomorrow / today - 1
pnl = signal * fwd - 0.0005
print("sharpe", pnl.mean() / pnl.std() * np.sqrt(252))

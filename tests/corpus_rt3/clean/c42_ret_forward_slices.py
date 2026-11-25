# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np
close = np.load("close.npy")
fwd = np.empty_like(close); fwd[:-1] = close[1:] / close[:-1] - 1; fwd[-1] = np.nan
signal = close > np.convolve(close, np.ones(50) / 50, mode="full")[: len(close)]
pnl = signal * fwd - 0.0005
print(np.nanmean(pnl) / np.nanstd(pnl) * np.sqrt(252))

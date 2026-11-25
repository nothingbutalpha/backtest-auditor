# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
px = pd.read_csv("btc.csv", index_col=0, parse_dates=True)["close"].to_numpy()
fast = pd.Series(px).ewm(span=12).mean().to_numpy()
slow = pd.Series(px).ewm(span=26).mean().to_numpy()
signal = np.where(fast > slow, 1.0, -1.0)
rets = np.diff(px) / px[:-1]
position = signal[:-1]
pnl = position * rets - 0.001 * np.abs(np.diff(np.concatenate([[0.0], position])))
print("sharpe", pnl.mean() / pnl.std() * np.sqrt(365))

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd, numpy as np
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
c = df["close"].values
sig = (c > pd.Series(c).rolling(20).mean().values).astype(int)
r = c[1:] / c[:-1] - 1
strat = sig[:-1] * r - 0.0005 * np.abs(np.diff(sig))
print("sharpe", strat.mean() / strat.std() * np.sqrt(252))

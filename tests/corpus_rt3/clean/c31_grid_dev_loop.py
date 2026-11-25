# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
dev = px.iloc[: int(len(px) * 0.7)]
hold = px.iloc[int(len(px) * 0.7):]
res = {}
for n in range(10, 200, 10):
    sig = (dev > dev.rolling(n).mean()).shift(1)
    r = sig * dev.pct_change() - 0.0005 * sig.diff().abs()
    res[n] = r.mean() / r.std() * np.sqrt(252)
best_n = max(res, key=res.get)
sig = (hold > hold.rolling(best_n).mean()).shift(1)
r = sig * hold.pct_change() - 0.0005 * sig.diff().abs()
print("holdout sharpe", r.mean() / r.std() * np.sqrt(252))

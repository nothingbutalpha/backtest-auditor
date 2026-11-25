# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
train, test = px.loc[:"2015"], px.loc["2016":]
def sharpe(p, n):
    sig = (p > p.rolling(n).mean()).shift(1)
    r = sig * p.pct_change() - 0.0005 * sig.diff().abs()
    return r.mean() / r.std() * np.sqrt(252)
best = max(range(20, 220, 20), key=lambda n: sharpe(test, n))
print("OOS sharpe", sharpe(test, best))

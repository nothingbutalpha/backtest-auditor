# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
thr = np.percentile(df["rsi"].dropna(), 80)
df["signal"] = (df["rsi"] > thr).astype(int)
df["strat"] = df["signal"].shift(1) * df["close"].pct_change() - 0.0005

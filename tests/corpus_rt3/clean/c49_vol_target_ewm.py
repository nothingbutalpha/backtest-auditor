# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd, numpy as np
r = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"].pct_change()
vol = r.ewm(span=60).std() * np.sqrt(252)
signal = (0.10 / vol).clip(upper=2)
pos = signal.shift(1)
strat = pos * r - 0.0005 * pos.diff().abs()

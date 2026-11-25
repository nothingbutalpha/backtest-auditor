# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
s = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["spread"]
z = (s - s.mean()) / s.std()
signal = (z < -2).astype(int)
pos = signal.shift(1)

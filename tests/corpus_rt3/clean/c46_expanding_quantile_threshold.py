# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
s = pd.read_csv("spread.csv", index_col=0, parse_dates=True)["spread"]
hi = s.expanding(252).quantile(0.9).shift(1)
signal = (s > hi).astype(int)
pos = signal.shift(1)

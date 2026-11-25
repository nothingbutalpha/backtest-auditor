# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
closes = pd.read_csv("closes.csv", index_col=0, parse_dates=True)
mom = closes.pct_change(252).shift(21)
signal = mom.rank(axis="columns", pct=True)
w = (signal > 0.8).astype(float)
w = w.div(w.sum(axis=1), axis=0).shift(1)

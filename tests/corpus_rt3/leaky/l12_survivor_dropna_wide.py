# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
closes = pd.read_csv("closes.csv", index_col=0, parse_dates=True).dropna(axis=1)
mom = closes.pct_change(126).shift(1)

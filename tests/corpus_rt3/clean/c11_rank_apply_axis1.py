# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
closes = pd.read_csv("closes.csv", index_col=0, parse_dates=True)
mom = closes.pct_change(126)
signal = mom.apply(lambda row: row.rank(pct=True), axis=1)
pos = (signal > 0.9).astype(int).shift(1)

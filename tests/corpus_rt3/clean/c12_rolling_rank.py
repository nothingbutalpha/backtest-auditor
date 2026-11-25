# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
s = pd.read_csv("vix.csv", index_col=0, parse_dates=True)["close"]
signal = s.rolling(252).rank(pct=True)
pos = (signal < 0.2).astype(int).shift(1)

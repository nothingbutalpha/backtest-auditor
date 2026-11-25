# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
closes = pd.read_csv("closes.csv", index_col=0, parse_dates=True)
keep = [c for c in closes if closes[c].last_valid_index() == closes.index[-1]]
closes = closes[keep]

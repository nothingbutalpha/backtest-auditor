# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
score = pd.read_csv("scores.csv", index_col=0, parse_dates=True)
signal = score.sub(score.mean(axis=1), axis=0)
signal = signal.div(signal.abs().max(axis=1), axis=0)
pos = signal.shift(1)

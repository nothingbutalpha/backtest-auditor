# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
latest = pd.read_csv("latest_scores.csv", index_col="ticker")["score"]
signal = latest / latest.abs().max()
weights = signal / signal.abs().sum()
weights.to_csv("target_weights.csv")

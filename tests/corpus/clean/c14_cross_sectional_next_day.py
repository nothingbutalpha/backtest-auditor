# expect: none
# The clean version of the Phase 0 reversal: rank on today's return, hold over tomorrow's return.
import pandas as pd

px = pd.read_parquet("closes.parquet")
returns = px.pct_change()
signal = (returns.rank(axis=1, ascending=True) <= 50).astype(float)
weights = signal.div(signal.sum(axis=1), axis=0).shift(1)
turnover = weights.diff().abs().sum(axis=1)
cost_bps = 10
strategy = (weights * returns).sum(axis=1) - turnover * cost_bps / 1e4

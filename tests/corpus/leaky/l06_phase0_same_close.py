# expect: LA07:high, CO01
# From our own Phase 0 experience: a cross-sectional short-term reversal ranked on today's return and
# credited with today's return. Signal and fill happen on the same close.
import pandas as pd

px = pd.read_parquet("closes.parquet")  # date x ticker closes
returns = px.pct_change()
signal = (returns.rank(axis=1, ascending=True) <= 50).astype(float)
weights = signal.div(signal.sum(axis=1), axis=0)
strategy = (weights * returns).sum(axis=1)
print("annualized Sharpe", strategy.mean() / strategy.std() * 252 ** 0.5)

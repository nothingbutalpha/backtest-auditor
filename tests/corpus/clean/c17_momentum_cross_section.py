# expect: none
import pandas as pd

px = pd.read_parquet("closes_pit_incl_delisted.parquet")
mom = px.pct_change(252).shift(21)
score = mom.rank(axis=1, pct=True)
long_book = (score >= 0.9).astype(float)
w = long_book.div(long_book.sum(axis=1), axis=0).shift(1)
daily = px.pct_change()
tcost = 0.001
pnl = (w * daily).sum(axis=1) - w.diff().abs().sum(axis=1) * tcost

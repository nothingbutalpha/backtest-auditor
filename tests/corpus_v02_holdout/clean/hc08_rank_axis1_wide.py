# expect: none
# Looks like a full-sample rank, but rank(axis=1) ranks across tickers within each date.
import numpy as np
import pandas as pd

prices = pd.read_csv("sector_etfs.csv", index_col="date", parse_dates=True)
rets = prices.pct_change()
mom = prices.shift(21) / prices.shift(252) - 1
rank = mom.rank(axis=1, ascending=False)

w = (rank <= 3).astype(float)
w = w.div(w.sum(axis=1), axis=0).fillna(0).shift(1).fillna(0)
turnover = w.diff().abs().sum(axis=1)
port = (w * rets).sum(axis=1) - 0.0005 * turnover
print("Sharpe", port.mean() / port.std() * np.sqrt(252))

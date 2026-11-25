# expect: none
import pandas as pd
closes = pd.read_csv("closes.csv", index_col=0, parse_dates=True)
mom = closes.pct_change(126)
ranks = mom.rank(axis=1, pct=True)
weights = (ranks > 0.8).astype(float)
weights = weights.div(weights.sum(axis=1), axis=0)
rets = closes.pct_change()
port = (weights.shift(1) * rets).sum(axis=1) - 0.001 * weights.diff().abs().sum(axis=1)

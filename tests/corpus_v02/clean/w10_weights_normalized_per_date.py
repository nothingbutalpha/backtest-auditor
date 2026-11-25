# expect: none
import pandas as pd
closes = pd.read_csv("closes.csv", index_col=0, parse_dates=True)
inv_vol = 1 / closes.pct_change().rolling(60).std()
weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
rets = closes.pct_change()
port = (weights.shift(1) * rets).sum(axis=1) - 0.0005 * weights.diff().abs().sum(axis=1)

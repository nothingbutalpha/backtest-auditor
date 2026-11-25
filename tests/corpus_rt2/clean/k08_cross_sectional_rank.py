import pandas as pd
px = pd.read_csv("wide.csv", index_col=0, parse_dates=True)
rets = px.pct_change()
mom = px.pct_change(252).shift(21)
rk = mom.rank(axis=1, pct=True)
weights = (rk > 0.8).astype(float)
weights = weights.div(weights.sum(axis=1), axis=0)
port = (weights.shift(1) * rets).sum(axis=1)
cost = weights.diff().abs().sum(axis=1) * 0.001
net = port - cost

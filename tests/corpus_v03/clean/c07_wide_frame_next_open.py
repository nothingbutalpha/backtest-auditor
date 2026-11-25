# expect: none
# Class 3: wide date x ticker frames, weights at t, returns from open t+1 to open t+2.
import pandas as pd

closes = pd.read_parquet("closes.parquet")
opens = pd.read_parquet("opens.parquet")
mom = closes.pct_change(20)
weights = mom.rank(axis=1, pct=True).gt(0.8).astype(float)
weights = weights.div(weights.sum(axis=1), axis=0)
px_in = opens.shift(-1)
px_out = opens.shift(-2)
rets = px_out / px_in - 1
gross = (weights * rets).sum(axis=1)
net = gross - 0.0005 * weights.diff().abs().sum(axis=1)
print(net.mean() * 252)

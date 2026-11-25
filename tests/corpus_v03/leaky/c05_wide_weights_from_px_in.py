# expect: LA01|LA10:high
# Twin of clean/c07: the ranking uses the next open.
import pandas as pd

closes = pd.read_parquet("closes.parquet")
opens = pd.read_parquet("opens.parquet")
px_in = opens.shift(-1)
px_out = opens.shift(-2)
gap = px_in / closes - 1
weights = gap.rank(axis=1, pct=True).gt(0.8).astype(float)
rets = px_out / px_in - 1
gross = (weights * rets).sum(axis=1) - 0.0005

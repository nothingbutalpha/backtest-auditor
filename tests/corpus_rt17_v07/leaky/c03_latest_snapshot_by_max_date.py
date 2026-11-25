# leak: 4
import pandas as pd
hist = pd.read_parquet("index_membership_snapshots.parquet")
newest = hist[hist["as_of"] == hist["as_of"].max()]
names = newest["ticker"].unique()
px = pd.read_parquet("all_closes.parquet")[names]
rets = px.pct_change()
sig = rets.rolling(252).sum().shift(21)
long = sig.rank(axis=1, ascending=False) <= 30
print(rets[long.shift(1).fillna(False)].mean(axis=1).cumsum().iloc[-1])

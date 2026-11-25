# expect: none
import pandas as pd
meta = pd.read_csv("security_attributes.csv", index_col="ticker").dropna(axis=1)
px = pd.read_parquet("all_closes.parquet")
rets = px.pct_change()
big = meta.index[meta["float_shares"] > 1e8]
w = rets[big.intersection(rets.columns)].rolling(20).mean().shift(1).gt(0).astype(float)
print((w * rets[w.columns]).mean(axis=1).sum())

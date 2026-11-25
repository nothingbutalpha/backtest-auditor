# expect: SB01@6
import pandas as pd

px = pd.read_parquet("tsx_close.parquet")
last_seen = px.apply(pd.Series.last_valid_index)
survivors = last_seen[last_seen == px.index[-1]].index
r = px[survivors].pct_change()
w = (r.rolling(126).sum().rank(axis=1, pct=True) > 0.9).astype(float).shift(1)
print((w.div(w.sum(axis=1), axis=0) * r).sum(axis=1).mean())

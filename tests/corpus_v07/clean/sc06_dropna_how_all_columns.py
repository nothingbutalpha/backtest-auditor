# expect: none
import pandas as pd

wide = pd.read_parquet("asx_close.parquet").dropna(axis=1, how="all")
r = wide.pct_change(fill_method=None)
w = (r.rolling(20).sum().rank(axis=1, pct=True) < 0.1).astype(float).shift(1)
print((w.div(w.sum(axis=1), axis=0) * r).sum(axis=1).mean())

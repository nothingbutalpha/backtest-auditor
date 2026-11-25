# expect: SB01:medium@5
import pandas as pd

px = pd.read_parquet("tsx_close.parquet").loc["2008":]
px = px.dropna(axis=1, thresh=len(px))
r = px.pct_change()
w = (r.rolling(20).sum().shift(1).rank(axis=1, pct=True) < 0.2).astype(float)
print((w.div(w.sum(axis=1), axis=0) * r).sum(axis=1).mean())

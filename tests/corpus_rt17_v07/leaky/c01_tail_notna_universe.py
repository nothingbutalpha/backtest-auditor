# leak: 4
import pandas as pd
px = pd.read_parquet("russell_closes.parquet")
alive = px.columns[px.tail(1).notna().all()]
rets = px[alive].pct_change()
w = rets.rolling(126).mean().rank(axis=1, pct=True).gt(0.9).astype(float)
w = w.div(w.sum(axis=1), axis=0).shift(1)
print((w * rets).sum(axis=1).add(1).prod())

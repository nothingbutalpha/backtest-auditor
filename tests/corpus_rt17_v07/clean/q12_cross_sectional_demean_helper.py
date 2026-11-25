# expect: none
import pandas as pd
def demean_rows(frame):
    return frame.sub(frame.mean(axis=1), axis=0)
px = pd.read_parquet("sector_etfs.parquet")
r = px.pct_change()
w = -demean_rows(r.rolling(5).sum()).shift(1)
w = w.div(w.abs().sum(axis=1), axis=0)
print((w * r).sum(axis=1).cumsum().iloc[-1])

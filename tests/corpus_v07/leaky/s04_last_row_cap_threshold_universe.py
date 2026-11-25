# expect: SB01@7
import pandas as pd

mcap = pd.read_parquet("eu_mcap.parquet")
close = pd.read_parquet("eu_close.parquet")
final_cap = mcap.ffill().iloc[-1]
names = final_cap[final_cap >= 5e9].index
r = close[names].pct_change()
w = (r.rolling(60).mean().rank(axis=1, pct=True) > 0.8).astype(float).shift(1)
print((w.div(w.sum(axis=1), axis=0) * r).sum(axis=1).mean() * 252)

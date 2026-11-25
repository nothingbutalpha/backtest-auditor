# expect: none
import pandas as pd
px = pd.read_parquet("all_closes.parquet")
vol = pd.read_parquet("all_volumes.parquet")
adv = (px * vol).rolling(20).mean().shift(1)
rets = px.pct_change()
ok = adv > 5e6
score = rets.rolling(60).sum().shift(1).where(ok)
w = score.rank(axis=1, pct=True).gt(0.8).astype(float)
w = w.div(w.sum(axis=1), axis=0)
print((w * rets).sum(axis=1).cumsum().iloc[-1])

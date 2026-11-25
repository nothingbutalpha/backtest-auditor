# expect: none
import pandas as pd
ref = pd.read_csv("security_master.csv")
sector_of = ref.groupby("ticker")["sector"].last()
px = pd.read_parquet("all_closes.parquet")
rets = px.pct_change()
w = rets.rolling(126).mean().rank(axis=1, pct=True).gt(0.9).astype(float).shift(1)
pnl = (w * rets).sum(axis=1)
exposure = w.T.groupby(sector_of).sum().T.mean()
print(pnl.cumsum().iloc[-1])
print(exposure.sort_values())

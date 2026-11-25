# expect: none
import pandas as pd
cap = pd.read_parquet("mktcap.parquet")
px = pd.read_parquet("all_closes.parquet").dropna(axis=1, how="all")
r = px.pct_change()
pnl = []
for d in r.index[21::21]:
    known = cap.loc[:d].iloc[-2]
    top = known.dropna().nlargest(100).index
    hold = r.loc[r.index > d, top].iloc[:21]
    pnl.append(hold.mean(axis=1))
print(pd.concat(pnl).sum())

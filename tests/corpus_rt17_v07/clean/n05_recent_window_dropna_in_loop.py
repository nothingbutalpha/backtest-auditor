# expect: none
import pandas as pd
wide = pd.read_parquet("eu_stocks.parquet")
ret = wide.pct_change()
pnl = []
for i in range(126, len(ret) - 21, 21):
    recent = ret.iloc[i - 126:i].dropna(axis=1)
    pick = recent.std().nsmallest(30).index
    pnl.append(ret.iloc[i:i + 21][pick].mean(axis=1))
print(pd.concat(pnl).cumsum().iloc[-1])

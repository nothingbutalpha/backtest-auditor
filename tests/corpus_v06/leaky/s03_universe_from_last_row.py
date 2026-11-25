# expect: SB01:medium@6
import pandas as pd

cap = pd.read_parquet("kospi_mcap.parquet")
rets = pd.read_parquet("kospi_rets.parquet")
names = cap.iloc[-1].nlargest(50).index
r = rets.loc["2012":, names]
w = (r.rolling(126).sum().shift(1).rank(axis=1, pct=True) > 0.7).astype(float)
print((w.div(w.sum(axis=1), axis=0) * r).sum(axis=1).mean())

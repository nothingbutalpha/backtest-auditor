# leak: 4
import pandas as pd
wide = pd.read_parquet("eu_stocks.parquet")
wide = wide.T.dropna().T
ret = wide.pct_change()
pos = ret.rolling(20).std().rank(axis=1, pct=True).lt(0.2).astype(float).shift(1)
print((pos * ret).sum(axis=1).cumsum().iloc[-1])

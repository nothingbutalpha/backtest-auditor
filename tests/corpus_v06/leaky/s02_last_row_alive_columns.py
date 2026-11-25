# expect: SB01:medium@5
import pandas as pd

px = pd.read_parquet("asx_all_listed.parquet")
px = px.loc[:, px.iloc[-1].notna()]
ret = px.pct_change(fill_method=None)
low_vol = ret.rolling(60).std().shift(1).rank(axis=1, pct=True) < 0.2
print(ret.where(low_vol).mean(axis=1).mean() * 252)

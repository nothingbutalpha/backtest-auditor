# expect: none
import pandas as pd

px = pd.read_parquet("sector_etfs.parquet")  # dates x tickers
mom = px.pct_change(126)
zmom = mom.T.apply(lambda c: (c - c.mean()) / c.std()).T
weights = (zmom > 0).astype(float).div((zmom > 0).sum(axis=1), axis=0).shift(1)
port = (weights * px.pct_change()).sum(axis=1) - 0.0005 * weights.diff().abs().sum(axis=1)
print(port.mean() * 252)

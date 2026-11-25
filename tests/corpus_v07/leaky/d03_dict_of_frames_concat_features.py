# expect: LA01|LA10@9
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

close = pd.read_parquet("vix_futures.parquet")["vx1"]
blocks = {}
blocks["chg_5"] = close.pct_change(5)
blocks["chg_20"] = close.pct_change(20)
blocks["chg_next"] = close.pct_change().shift(-1)
X = pd.concat(blocks, axis=1).dropna()
y = close.pct_change(5).shift(-5).reindex(X.index)
gb = GradientBoostingRegressor().fit(X.loc[:"2019"], y.loc[:"2019"].fillna(0))
print(gb.score(X.loc["2020":], y.loc["2020":].fillna(0)))

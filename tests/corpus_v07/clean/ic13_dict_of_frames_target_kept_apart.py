# expect: none
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

close = pd.read_parquet("vix_futures.parquet")["vx1"]
blocks = {"chg_5": close.pct_change(5), "chg_20": close.pct_change(20)}
X = pd.concat(blocks, axis=1).dropna()
y = close.pct_change(5).shift(-5).reindex(X.index)
gb = GradientBoostingRegressor().fit(X.loc[:"2019"].iloc[:-5], y.loc[:"2019"].iloc[:-5])
print(gb.score(X.loc["2020":].iloc[:-5], y.loc["2020":].iloc[:-5]))

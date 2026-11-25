# expect: none
# LA05 class: a closed date window for fitting, later dates for prediction.
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

X = pd.read_parquet("X.parquet")
y = pd.read_parquet("y.parquet")["fwd_ret"]
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=2)
gbr.fit(X.loc["2010-01-01":"2017-12-31"], y.loc["2010-01-01":"2017-12-31"])
pred = pd.Series(gbr.predict(X.loc["2018-01-01":]), index=X.loc["2018-01-01":].index)

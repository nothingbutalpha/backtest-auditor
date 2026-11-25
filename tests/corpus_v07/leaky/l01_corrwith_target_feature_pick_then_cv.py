# expect: LA05|LA11|LA14@9
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge

data = pd.read_parquet("credit_macro.parquet")
y = data.pop("hy_spread_fwd_3m")
ic = data.corrwith(y).abs()
chosen = ic.nlargest(10).index
print(cross_val_score(Ridge(), data[chosen], y, cv=TimeSeriesSplit(5)).mean())

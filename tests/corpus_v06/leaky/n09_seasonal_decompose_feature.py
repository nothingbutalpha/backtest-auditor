# expect: LA02:high@8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

h = pd.read_parquet("power_hourly.parquet")
parts = seasonal_decompose(h["load"], period=24, extrapolate_trend="freq")
h["load_trend"] = parts.trend
h["y"] = h["price"].shift(-1)
data = h.dropna()
rf = RandomForestRegressor(n_estimators=200).fit(data[["load_trend"]].iloc[:8000], data["y"].iloc[:8000])
print(rf.score(data[["load_trend"]].iloc[8000:], data["y"].iloc[8000:]))

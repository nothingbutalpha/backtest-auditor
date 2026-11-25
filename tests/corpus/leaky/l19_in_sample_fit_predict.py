# expect: LA05:medium
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("features.csv", parse_dates=["date"], index_col="date")
X = df[["mom", "value", "quality"]]
y = df["fwd_ret_21d"]
model = GradientBoostingRegressor()
model.fit(X, y)
df["pred"] = model.predict(X)
df["position"] = (df["pred"] > 0).astype(int)

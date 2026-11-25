# expect: none
# Class 1: drop chained with select_dtypes / fillna.
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_parquet("panel.parquet")
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-10) / df["close"] - 1
df["y"] = df["fwd_ret"].clip(-0.5, 0.5)
df["mom"] = df.groupby("ticker")["close"].pct_change(126)
df["size"] = df["mktcap"].rank(pct=True)
X = df.drop(columns=["fwd_ret", "y"]).select_dtypes("number").fillna(0)
cut = df["date"] < "2021-01-01"
hgb = HistGradientBoostingRegressor().fit(X[cut], df.loc[cut, "y"])

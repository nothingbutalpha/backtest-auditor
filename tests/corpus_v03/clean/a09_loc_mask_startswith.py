# expect: none
# Class 1: forward columns removed with a boolean column mask.
import pandas as pd
from sklearn.linear_model import Lasso

df = pd.read_parquet("panel.parquet")
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["fwd_vol"] = df.groupby("ticker")["close"].pct_change().shift(-5).rolling(5).std()
df["mom"] = df.groupby("ticker")["close"].pct_change(20)
df = df.dropna()
X = df.loc[:, ~df.columns.str.startswith("fwd_")].drop(columns=["date", "ticker"])
y = df["fwd_ret"]
cut = df["date"] < "2020-01-01"
Lasso(alpha=1e-4).fit(X[cut], y[cut])

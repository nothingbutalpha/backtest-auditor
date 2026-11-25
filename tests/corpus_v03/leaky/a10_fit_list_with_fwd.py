# expect: LA10:high
# A forward column passed straight to fit.
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_parquet("panel.parquet")
df["fwd_close"] = df.groupby("ticker")["close"].shift(-1)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["mom"] = df.groupby("ticker")["close"].pct_change(20)
df = df.dropna()
cut = df["date"] < "2020-01-01"
LinearRegression().fit(df.loc[cut, ["mom", "fwd_close"]], df.loc[cut, "fwd_ret"])

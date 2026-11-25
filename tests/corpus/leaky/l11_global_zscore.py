# expect: LA09:medium
import numpy as np
import pandas as pd

df = pd.read_csv("pair.csv", parse_dates=["date"], index_col="date")
df["spread"] = np.log(df["close_a"]) - np.log(df["close_b"])
df["z"] = (df["spread"] - df["spread"].mean()) / df["spread"].std()
df["position"] = np.where(df["z"] > 2, -1, np.where(df["z"] < -2, 1, 0))
df["position"] = df["position"].shift(1)
df["spread_ret"] = df["spread"].diff()
fees = 0.0002
df["pnl"] = df["position"] * df["spread_ret"] - df["position"].diff().abs() * fees

# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["ret"] = np.log(df["close"] / df["close"].shift(1))
df["position"] = np.sign(df["ret"].rolling(20).mean()).shift(1)
df["turnover"] = df["position"].diff().abs()
tc = 0.0005
df["strategy"] = df["position"] * df["ret"] - df["turnover"] * tc

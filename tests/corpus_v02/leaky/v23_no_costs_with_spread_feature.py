# expect: CO01:possible
import numpy as np
import pandas as pd
df = pd.read_csv("pairs.csv", index_col=0, parse_dates=True)
df["spread"] = np.log(df["ko"]) - np.log(df["pep"])
df["z"] = (df["spread"] - df["spread"].rolling(60).mean()) / df["spread"].rolling(60).std()
df["position"] = -np.sign(df["z"]).shift(1)
df["pair_ret"] = df["ko"].pct_change() - df["pep"].pct_change()
df["strategy"] = df["position"] * df["pair_ret"]
print(df["strategy"].cumsum().iloc[-1], df["strategy"].mean() / df["strategy"].std() * np.sqrt(252))

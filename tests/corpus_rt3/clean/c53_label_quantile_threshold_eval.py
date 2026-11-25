# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("features.parquet")
df["target"] = (df["fwd_ret"] > df["fwd_ret"].quantile(0.7)).astype(int)
train, test = df.loc[:"2018"], df.loc["2019":]

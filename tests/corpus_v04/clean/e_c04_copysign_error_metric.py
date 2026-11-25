# expect: none
import numpy as np
import pandas as pd

df = pd.read_parquet("predictions.parquet")
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
signed_err = np.copysign(np.abs(df["pred"] - df["fwd_ret"]), df["fwd_ret"])
print(signed_err.mean(), signed_err.abs().mean())

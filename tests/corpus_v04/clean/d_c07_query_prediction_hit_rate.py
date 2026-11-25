# expect: none
import pandas as pd

df = pd.read_parquet("predictions.parquet")
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
hit_rate = df.query("pred > 0")["fwd_ret"].gt(0).mean()
print(hit_rate)

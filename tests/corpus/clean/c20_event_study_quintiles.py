# expect: none
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_parquet("signals.parquet")
df["fwd_ret_5"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
summary = df.groupby(df.groupby("date")["score"].transform(lambda s: pd.qcut(s, 5, labels=False)))["fwd_ret_5"].mean()
print(summary)

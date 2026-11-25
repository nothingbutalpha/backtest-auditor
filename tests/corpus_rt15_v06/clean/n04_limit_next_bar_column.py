# expect: none
import pandas as pd
df = pd.read_parquet("nq_1m.parquet")
df["limit_price"] = df["close"] - 2.0
df["low_next"] = df["low"].shift(-1)
df["got_filled"] = df["low_next"] <= df["limit_price"]
df["pnl"] = (df["close"].shift(-5) - df["limit_price"]).where(df["got_filled"], 0.0)
print(df["pnl"].sum())

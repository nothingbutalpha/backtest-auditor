# leak: 5
import pandas as pd
df = pd.read_parquet("nq_1m.parquet")
df["limit_price"] = df.close - 2.0
df["got_filled"] = df.low <= df.limit_price
df["pnl"] = (df.close.shift(-5) - df.limit_price).where(df.got_filled, 0.0)
print(df.pnl.sum())

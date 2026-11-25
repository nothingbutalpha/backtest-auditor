# expect: none
# Class 2: quintile returns via pivot_table on the forward return.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["q"] = df.groupby("date")["earnings_yield"].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
tbl = df.pivot_table(index="date", columns="q", values="fwd_ret", aggfunc="mean")
q_spread = tbl[4] - tbl[0]
print(tbl.mean(), q_spread.mean())

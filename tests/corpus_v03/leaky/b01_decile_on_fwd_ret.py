# expect: LA10
# Twin of clean/b01: deciles are formed on the forward return itself, so the spread is guaranteed.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
g = df.groupby("ticker")["close"]
df["fwd_ret"] = g.shift(-21) / df["close"] - 1
df["decile"] = df.groupby("date")["fwd_ret"].transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
by_decile = df.groupby(["date", "decile"])["fwd_ret"].mean().unstack()
long_short = by_decile[9] - by_decile[0]
print(long_short.mean() / long_short.std() * (12 ** 0.5))

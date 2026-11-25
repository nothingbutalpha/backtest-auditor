# expect: LA10:high
# Twin of clean/b04: the long leg is picked by ranking on the forward return.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
px = df.groupby("ticker")["close"]
df["fwd_ret"] = px.shift(-21) / df["close"] - 1
df["score"] = df.groupby("date")["fwd_ret"].rank(pct=True)
long_leg = df[df["score"] > 0.9].groupby("date")["fwd_ret"].mean()
print(long_leg.mean())

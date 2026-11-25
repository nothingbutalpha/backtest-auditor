# expect: LA10:high
# Twin of clean/b05: weights are ranked on the forward return.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
c = df.groupby("ticker")["close"]
df["fwd_ret"] = c.shift(-1) / df["close"] - 1
df["weight"] = df.groupby("date")["fwd_ret"].rank(pct=True) - 0.5
df["contrib"] = df["weight"] * df["fwd_ret"]
port_ret = df.groupby("date")["contrib"].sum() - 0.0005
print(port_ret.mean() * 252)

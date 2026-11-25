# expect: none
# Class 2: weights decided at t times the return from t to t+1, summed per date.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
c = df.groupby("ticker")["close"]
df["mom"] = c.pct_change(60)
df["fwd_ret"] = c.shift(-1) / df["close"] - 1
df["weight"] = df.groupby("date")["mom"].rank(pct=True) - 0.5
df["weight"] = df["weight"] / df.groupby("date")["weight"].transform(lambda w: w.abs().sum())
df["contrib"] = df["weight"] * df["fwd_ret"]
port_ret = df.groupby("date")["contrib"].sum()
turnover = df.groupby("ticker")["weight"].diff().abs().groupby(df["date"]).sum()
net_ret = port_ret - 0.0005 * turnover
print(net_ret.mean() * 252)

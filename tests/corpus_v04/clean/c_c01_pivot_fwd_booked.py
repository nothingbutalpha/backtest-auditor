# expect: none
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
df["mom"] = df.groupby("ticker")["close"].pct_change(120)
mom = df.pivot(index="date", columns="ticker", values="mom")
fwd = df.pivot(index="date", columns="ticker", values="fwd_ret")
weights = (mom.rank(axis=1, pct=True) > 0.8).astype(float)
weights = weights.div(weights.sum(axis=1), axis=0)
port = (weights * fwd).sum(axis=1) - 0.0005 * weights.diff().abs().sum(axis=1)
print(port.mean() * 252)

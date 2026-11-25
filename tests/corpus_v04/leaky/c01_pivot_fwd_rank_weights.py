# expect: LA10|LA01:high
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
fwd = df.pivot(index="date", columns="ticker", values="fwd_ret")
best = fwd.rank(axis=1, ascending=False) <= 20
weights = best.astype(float).div(best.sum(axis=1), axis=0)
port = (weights * rets.shift(-1)).sum(axis=1) - 0.001
print(port.mean() * 252)

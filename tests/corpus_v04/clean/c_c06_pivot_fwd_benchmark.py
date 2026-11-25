# expect: none
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
df["mom"] = df.groupby("ticker")["close"].pct_change(120)
fwd = df.pivot(index="date", columns="ticker", values="fwd_ret")
mom = df.pivot(index="date", columns="ticker", values="mom")
w = (mom.rank(axis=1, pct=True) > 0.8).astype(float)
port = (w * fwd).sum(axis=1) / w.sum(axis=1)
ew = fwd.mean(axis=1)
excess = port - ew
print(excess.mean() * 252)

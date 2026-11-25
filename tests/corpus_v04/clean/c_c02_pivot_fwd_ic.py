# expect: none
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
df["mom"] = df.groupby("ticker")["close"].pct_change(120)
mom_w = df.pivot(index="date", columns="ticker", values="mom")
fwd_w = df.pivot(index="date", columns="ticker", values="fwd_ret")
ic = mom_w.corrwith(fwd_w, axis=1, method="spearman")
print(ic.mean(), ic.mean() / ic.std() * len(ic) ** 0.5)

# expect: LA10|LA01:high
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
tbl = df.pivot_table("fwd_ret", index="date", columns="ticker")
signal = tbl.gt(tbl.median(axis=1), axis=0).astype(int)
strat = (signal * rets.shift(-1)).mean(axis=1) - 0.0005
print(strat.mean() * 252)

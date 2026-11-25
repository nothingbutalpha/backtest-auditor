# expect: none
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
df["mom"] = df.groupby("ticker")["close"].pct_change(120)
fw = df.set_index(["date", "ticker"])["fwd_ret"].unstack()
q = df.set_index(["date", "ticker"])["mom"].unstack().rank(axis=1, pct=True)
spread = fw[q > 0.9].mean(axis=1) - fw[q < 0.1].mean(axis=1)
print(spread.mean() * 252, spread.std() * 252 ** 0.5)

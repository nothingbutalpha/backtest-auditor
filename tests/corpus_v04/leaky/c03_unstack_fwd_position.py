# expect: LA10|LA01:high
import numpy as np
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
fw = df.set_index(["date", "ticker"])["fwd_ret"].unstack("ticker")
position = np.sign(fw.sub(fw.mean(axis=1), axis=0))
pnl = (position * rets.shift(-1)).sum(axis=1) / position.abs().sum(axis=1)
print(pnl.mean() * 252)

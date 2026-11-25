# expect: LA10|LA01:high
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["fwd_5d"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
df["ret"] = df.groupby("ticker")["close"].pct_change()
rets = df.pivot(index="date", columns="ticker", values="ret")
fw = df.pivot(index="date", columns="ticker", values="fwd_5d")
mom = fw.shift(1)
signal = (mom.rank(axis=1, pct=True) > 0.9).astype(float)
print((signal * rets).sum(axis=1).div(signal.sum(axis=1)).mean())

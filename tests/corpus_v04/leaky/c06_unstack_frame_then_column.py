# expect: LA10|LA01:high
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
rets = df.pivot(index="date", columns="ticker", values="ret")
wide = df.set_index(["date", "ticker"])[["ret", "fwd_ret"]].unstack("ticker")
fwide = wide["fwd_ret"]
score = fwide.rank(axis=1, pct=True)
weights = (score > 0.8).astype(float)
print((weights * rets.shift(-1)).sum(axis=1).div(weights.sum(axis=1)).mean())

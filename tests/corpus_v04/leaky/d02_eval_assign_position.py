# expect: LA10|LA01:high
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["fwd_5d"] = df["close"].shift(-5) / df["close"] - 1
df = df.eval("position = fwd_5d > 0.01")
df["strat"] = df["position"].shift(1) * df["ret"]
print(df["strat"].mean() * 252)

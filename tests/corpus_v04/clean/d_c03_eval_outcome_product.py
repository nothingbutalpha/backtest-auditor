# expect: none
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df["mom"] = df["close"].pct_change(20).shift(1)
df["position"] = (df["mom"] > 0).astype(int)
df["strat"] = df.eval("position * fwd_ret")
print(df["strat"].mean() * 252)

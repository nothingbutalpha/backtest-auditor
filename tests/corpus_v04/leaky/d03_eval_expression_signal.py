# expect: LA10|LA01:high
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df["mom"] = df["close"].pct_change(20).shift(1)
df["signal"] = df.eval("fwd_ret > mom").astype(int)
df["pnl"] = df["signal"] * df["fwd_ret"]
print(df["pnl"].sum())

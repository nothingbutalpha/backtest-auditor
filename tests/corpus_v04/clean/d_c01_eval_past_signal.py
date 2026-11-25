# expect: none
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df["mom"] = df["close"].pct_change(20).shift(1)
df.eval("signal = mom > 0", inplace=True)
df["pnl"] = df["signal"] * df["fwd_ret"] - 0.0005
print(df["pnl"].mean() * 252)

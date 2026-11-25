# expect: none
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df["mom"] = df["close"].pct_change(20).shift(1)
sig_days = df.query("mom > 0.05").index
df["position"] = df.index.isin(sig_days).astype(int)
df["pnl"] = df["position"] * df["fwd_ret"] - 0.0005 * df["position"].diff().abs()
print(df["pnl"].mean() * 252)

# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["fwd_ret"] = df["close"].pct_change(5).shift(-5)
df["mom"] = df["close"].pct_change(20)
df["mom_bucket"] = df["mom"].rolling(252).rank(pct=True)
summary = df.groupby(pd.cut(df["mom_bucket"], 5))["fwd_ret"].mean()
print(summary)

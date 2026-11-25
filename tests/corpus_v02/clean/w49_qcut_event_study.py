# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(20)
df["fwd_ret"] = df["close"].pct_change(5).shift(-5)
df["bucket"] = pd.qcut(df["mom"], 5, labels=False)
print(df.groupby("bucket")["fwd_ret"].agg(["mean", "count"]))

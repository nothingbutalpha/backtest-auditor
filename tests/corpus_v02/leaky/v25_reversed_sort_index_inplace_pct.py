# expect: LA02:high
import pandas as pd
df = pd.read_csv("prices.csv", index_col="date", parse_dates=True)
df.sort_index(ascending=False, inplace=True)
df["ret"] = df["close"].pct_change()
df["mom"] = df["close"].pct_change(20)
df["signal"] = (df["mom"] > 0).astype(int)

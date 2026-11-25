# expect: none
import pandas as pd
df = pd.read_csv("prices.csv", parse_dates=["date"])
df = df.sort_values("date", ascending=False).drop_duplicates("date")
df = df.sort_values("date")
df["ret"] = df["close"].pct_change()
df["vol"] = df["ret"].rolling(20).std()

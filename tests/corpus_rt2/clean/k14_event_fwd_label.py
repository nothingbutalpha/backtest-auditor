import pandas as pd
df = pd.read_csv("events.csv", parse_dates=["date"])
df["fwd_ret_5d"] = df.groupby("ticker")["close"].pct_change(5).shift(-5)
summary = df.groupby("event_type")["fwd_ret_5d"].mean()
print(summary)

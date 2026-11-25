# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
df["mom"] = df.groupby("ticker")["close"].pct_change(126)
last_seen = df.groupby("ticker")["date"].transform("max")
today = df["date"].max()
live = df[last_seen == today]
orders = live[live["date"] == today].nlargest(20, "mom")
orders.to_csv("orders_today.csv")

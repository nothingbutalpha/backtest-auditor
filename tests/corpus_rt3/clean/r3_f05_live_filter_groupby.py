# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["mom"] = df.groupby("ticker")["close"].pct_change(126)
today = df["date"].max()
tradable = df.groupby("ticker").filter(lambda g: g["date"].max() == today)
orders = tradable[tradable["date"] == today].nlargest(20, "mom")
orders.to_csv("orders.csv")

# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("spy.csv", parse_dates=["date"]).sort_values(by="date", ascending=0)
df["ret"] = df["close"].pct_change()
df["signal"] = df["ret"].rolling(10).mean() > 0

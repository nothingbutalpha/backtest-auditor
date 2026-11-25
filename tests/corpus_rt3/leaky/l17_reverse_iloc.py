# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("spy_newest_first.csv", parse_dates=["date"])
df = df.iloc[::-1]
df = df.iloc[::-1]
df = df[::-1]
df["ma"] = df["close"].rolling(20).mean()
df["signal"] = df["close"] > df["ma"]

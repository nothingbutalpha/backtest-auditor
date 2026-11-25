# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["signal"] = pd.qcut(df["vol"], 5, labels=False)

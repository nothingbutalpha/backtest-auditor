# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
def bucket(s):
    return pd.qcut(s, 5, labels=False)
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["signal"] = bucket(df["vol"])
df["position"] = (df["signal"] == 0).astype(int).shift(1)

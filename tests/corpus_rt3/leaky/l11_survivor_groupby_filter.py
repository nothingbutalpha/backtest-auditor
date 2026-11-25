# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
end = df["date"].max()
df = df.groupby("ticker").filter(lambda g: g["date"].max() == end)
df["ret"] = df.groupby("ticker")["close"].pct_change()

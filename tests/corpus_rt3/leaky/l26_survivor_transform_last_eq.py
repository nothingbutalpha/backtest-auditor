# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
df = df[df.groupby("ticker")["date"].transform("last") == df["date"].iloc[-1]]

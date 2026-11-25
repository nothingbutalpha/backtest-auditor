# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
n = df["date"].nunique()
counts = df.groupby("ticker")["date"].count()
df = df[df["ticker"].isin(counts[counts == n].index)]

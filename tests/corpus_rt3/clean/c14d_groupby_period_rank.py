# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("monthly.parquet")
df["signal"] = df.groupby("month")["bm"].rank(pct=True)
df["position"] = df.groupby("permno")["signal"].shift(1)

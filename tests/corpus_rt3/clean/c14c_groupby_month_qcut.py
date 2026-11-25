# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("monthly.parquet")
df["signal"] = df.groupby("yyyymm")["bm"].transform(lambda x: pd.qcut(x, 10, labels=False))
df["position"] = df.groupby("permno")["signal"].shift(1)

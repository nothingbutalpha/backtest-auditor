# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("crsp.parquet")
last = df.groupby("permno")["date"].transform("max")
end = df["date"].max()
df["delisted_flag"] = ~(last == end)
df.loc[df["delisted_flag"] & (df["date"] == last), "ret"] = df["dlret"]

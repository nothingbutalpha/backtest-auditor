# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("crsp.parquet")
last = df.groupby("permno")["date"].max()
end = df["date"].max()
censored = last[last == end].index
events = last[last < end]
print(len(censored), "still listed;", len(events), "delistings")
df["delisted"] = ~df["permno"].isin(censored)

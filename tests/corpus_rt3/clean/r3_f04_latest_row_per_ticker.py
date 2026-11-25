# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
latest = df.sort_values("date", ascending=False).groupby("ticker").head(1)
latest.to_csv("latest_quotes.csv", index=False)

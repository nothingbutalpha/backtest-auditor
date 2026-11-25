# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
df["rank_in_train"] = df.loc[:"2018", "mom"].rank(pct=True)

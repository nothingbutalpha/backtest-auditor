# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet")
alive = df.loc[df["date"] == df["date"].max(), "ticker"].unique()
df = df[df["ticker"].isin(alive)]
df["ret"] = df.groupby("ticker")["close"].pct_change()

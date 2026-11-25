# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
df["mom"] = df.groupby("ticker")["close"].pct_change(126)
df["signal"] = df.groupby("date")["mom"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
df["position"] = df.groupby("ticker")["signal"].shift(1)

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_parquet("panel.parquet").set_index("date").sort_index()
df["mom"] = df.groupby("ticker")["close"].pct_change(126)
df["signal"] = df.groupby(df.index)["mom"].rank(pct=True)
df["position"] = df.groupby("ticker")["signal"].shift(1)

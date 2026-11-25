# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
ticks = pd.read_parquet("ticks.parquet")
opts = {"label": "left", "closed": "left"}
bars = ticks["price"].resample("5min", **opts).ohlc()
ticks = ticks.join(bars["close"].rename("bar_close"))
ticks["signal"] = ticks["price"] > ticks["bar_close"].ffill()

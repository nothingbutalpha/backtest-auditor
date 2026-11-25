# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
ticks = pd.read_parquet("ticks.parquet")
kw = dict(label="right", closed="right")
bars = ticks["price"].resample("5min", **kw).ohlc()
ticks = ticks.join(bars["close"].rename("bar_close"))

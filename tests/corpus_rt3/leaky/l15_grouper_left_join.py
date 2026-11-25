# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
ticks = pd.read_parquet("ticks.parquet")
bars = ticks.groupby(pd.Grouper(freq="5min"))["price"].ohlc()
ticks = ticks.join(bars["close"].rename("bar_close"))
ticks["bar_close"] = ticks["bar_close"].ffill()
ticks["signal"] = ticks["price"] > ticks["bar_close"]

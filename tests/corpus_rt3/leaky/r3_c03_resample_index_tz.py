# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
m5 = pd.read_parquet("eurusd_5m.parquet")
h = m5.resample("1h").agg({"high": "max", "close": "last"})
h.index = h.index.tz_convert("UTC")
m5 = m5.join(h.add_prefix("h_")).ffill()
m5["signal"] = m5["close"] > m5["h_high"]

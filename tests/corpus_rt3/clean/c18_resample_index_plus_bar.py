# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
m5 = pd.read_parquet("eurusd_5m.parquet")
h = m5.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
h.index = h.index + pd.Timedelta(hours=1)
m5 = m5.join(h.add_prefix("h_"), how="left").ffill()
m5["signal"] = m5["close"] > m5["h_high"]

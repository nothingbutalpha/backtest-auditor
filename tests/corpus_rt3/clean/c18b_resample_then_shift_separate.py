# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
m5 = pd.read_parquet("eurusd_5m.parquet")
h = m5["close"].resample("1h").last()
h = h.shift(1)
m5["prev_h"] = h.reindex(m5.index, method="ffill")

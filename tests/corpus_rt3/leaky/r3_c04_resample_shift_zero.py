# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
m1 = pd.read_parquet("es_1min.parquet")
h = m1["close"].resample("1h").max()
h = h.shift(0)
m1["h_max"] = h.reindex(m1.index, method="ffill")
m1["signal"] = m1["close"] >= m1["h_max"]

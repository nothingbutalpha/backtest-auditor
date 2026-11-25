# expect: none
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
m1 = pd.read_parquet("es_1min.parquet")
h = m1["close"].resample("1h", label="right", closed="right").last()
m1["h"] = h.reindex(m1.index, method="ffill")

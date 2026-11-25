# expect: any
# From the second red-team review of PR #8 (5f3ea0a), 2026-10-16.
import pandas as pd
m1 = pd.read_parquet("es_1min.parquet")
hourly = m1["close"].resample("1h").last().shift(-1)
m1 = m1.join(hourly.rename("h_close"))
m1["signal"] = m1["close"] > m1["h_close"].ffill()

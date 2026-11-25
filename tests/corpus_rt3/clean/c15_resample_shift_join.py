# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
m1 = pd.read_parquet("es_1min.parquet")
hourly = m1["close"].resample("1h").last().shift(1)
m1 = m1.join(hourly.rename("prev_hour_close"), how="left")
m1["prev_hour_close"] = m1["prev_hour_close"].ffill()
m1["signal"] = m1["close"] > m1["prev_hour_close"]

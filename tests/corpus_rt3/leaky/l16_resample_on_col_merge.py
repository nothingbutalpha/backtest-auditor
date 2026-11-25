# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
m1 = pd.read_csv("m1.csv", parse_dates=["ts"])
h = m1.resample("1h", on="ts")["close"].last().reset_index()
m1 = pd.merge_asof(m1.sort_values("ts"), h.rename(columns={"close": "h_close"}), on="ts")
m1["signal"] = m1["close"] > m1["h_close"]

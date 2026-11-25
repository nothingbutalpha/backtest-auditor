# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
h = pd.read_parquet("btc_1h.parquet")
d = h["close"].resample("1D").last()
sig = (d > d.rolling(20).mean()).shift(1)
h["signal"] = sig.reindex(h.index, method="ffill")

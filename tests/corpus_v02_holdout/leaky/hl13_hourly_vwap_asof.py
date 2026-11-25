# expect: LA12
# Hourly VWAP bars are labelled at the hour start and merged with merge_asof onto tick timestamps.
import pandas as pd

ticks = pd.read_parquet("eurusd_ticks.parquet").sort_values("ts")
t = ticks.set_index("ts")
pv = (t["price"] * t["size"]).resample("1h").sum()
vv = t["size"].resample("1h").sum()
hourly = (pv / vv).rename("vwap_1h").reset_index()

ticks = pd.merge_asof(ticks, hourly, on="ts", direction="backward")
ticks["side"] = (ticks["price"] < ticks["vwap_1h"]).map({True: 1, False: -1})
ticks["pnl"] = ticks["side"].shift(1) * ticks["price"].diff() - 0.00001 * ticks["side"].diff().abs()
print(ticks["pnl"].sum())

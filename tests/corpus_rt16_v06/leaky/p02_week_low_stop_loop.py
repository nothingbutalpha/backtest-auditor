# leak: 6
import pandas as pd
h = pd.read_parquet("eth_1h.parquet")
h["entry_px"] = h["open"]
wk = h.index.to_period("W")
h["stop_px"] = h.groupby(wk)["low"].transform("min") * 1.001
pnl = []
for _, row in h.iterrows():
    exit_px = row["stop_px"] if row["low"] <= row["stop_px"] else row["close"]
    pnl.append(exit_px / row["entry_px"] - 1)
print(sum(pnl))

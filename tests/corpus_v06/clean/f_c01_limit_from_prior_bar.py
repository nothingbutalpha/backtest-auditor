# expect: none
import numpy as np
import pandas as pd

bars = pd.read_parquet("matic_5m.parquet")
bars["limit"] = bars["close"].shift(1) * 0.997
bars["hit"] = bars["low"] <= bars["limit"]
bars["fill_px"] = np.minimum(bars["limit"], bars["open"])
bars["exit_px"] = bars["close"].shift(-6)
t = bars[bars["hit"]].dropna(subset=["exit_px"])
print((t["exit_px"] / t["fill_px"] - 1).mean())

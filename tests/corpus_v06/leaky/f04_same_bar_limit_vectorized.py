# expect: LA07:high@6
import pandas as pd

bars = pd.read_parquet("matic_5m.parquet")
bars["limit"] = bars["close"] * 0.997
bars["hit"] = bars["low"] <= bars["limit"]
bars["exit_px"] = bars["close"].shift(-6)
t = bars[bars["hit"]].dropna(subset=["exit_px"])
print((t["exit_px"] / t["limit"] - 1).mean())

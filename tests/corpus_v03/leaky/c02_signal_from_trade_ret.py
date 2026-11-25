# expect: LA01|LA10:high
# Twin of clean/c02: the position is ranked on the open-to-open trade return, which ends at t+2.
import pandas as pd

df = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
g = df.groupby("ticker")
df["entry_px"] = g["open"].shift(-1)
df["exit_px"] = g["open"].shift(-2)
df["oo_ret"] = df["exit_px"] / df["entry_px"] - 1
df["position"] = (df.groupby("date")["oo_ret"].rank(pct=True) > 0.9).astype(float)
df["pnl"] = df["position"] * df["oo_ret"] - 0.001

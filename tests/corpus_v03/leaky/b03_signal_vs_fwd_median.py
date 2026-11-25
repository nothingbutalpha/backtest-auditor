# expect: LA10:high
# Twin of clean/b03: the position is chosen from the forward return.
import pandas as pd

panel = pd.read_csv("panel.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
close = panel.groupby("ticker")["close"]
panel["fwd_ret"] = close.shift(-1) / panel["close"] - 1
panel["signal"] = (panel["fwd_ret"] > panel.groupby("date")["fwd_ret"].transform("median")).astype(int)
panel["pnl"] = panel["signal"] * panel["fwd_ret"] - 0.001
print(panel.groupby("date")["pnl"].mean().mean() * 252)

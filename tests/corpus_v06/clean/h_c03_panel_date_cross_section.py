# expect: none
import pandas as pd

panel = pd.read_parquet("panel_daily.parquet")
panel["xs_mean"] = panel.groupby("date")["ret_20"].transform("mean")
panel["rel"] = panel["ret_20"] - panel["xs_mean"]
panel["signal"] = (panel["rel"] > 0).astype(float)
panel["fwd"] = panel.groupby("ticker")["close"].shift(-1) / panel["close"] - 1
print((panel["signal"] * panel["fwd"]).groupby(panel["date"]).mean().mean())

# expect: none
import pandas as pd

panel = pd.read_parquet("crypto_panel.parquet").sort_values(["symbol", "date"])
panel["ret"] = panel.groupby("symbol")["close"].pct_change()
g = panel.groupby("symbol")
panel["vol_chg"] = g["volume"].pct_change().groupby(panel["symbol"]).shift(1)
panel["signal"] = (panel["vol_chg"] > 1.0).astype(int)
pnl = (panel["signal"] * panel["ret"]).groupby(panel["date"]).mean()
print(pnl.mean())

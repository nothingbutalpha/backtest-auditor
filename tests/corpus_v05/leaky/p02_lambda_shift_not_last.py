# expect: LA07:high
import pandas as pd

panel = pd.read_parquet("crypto_panel.parquet").sort_values(["symbol", "date"])
panel["ret"] = panel.groupby("symbol")["close"].pct_change()
panel["gap"] = panel.groupby("symbol")["close"].transform(lambda s: s / s.shift(1) - 1)
panel["signal"] = (panel["gap"] > 0.05).astype(int)
pnl = (panel["signal"] * panel["ret"]).groupby(panel["date"]).mean()
print(pnl.mean())

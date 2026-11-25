# expect: LA07:high
import pandas as pd

panel = pd.read_parquet("crypto_panel.parquet").sort_values(["symbol", "date"])
panel["ret"] = panel.groupby("symbol")["close"].pct_change()
panel["mom_7"] = panel.groupby("symbol")["close"].transform(lambda s: s.pct_change(7))
panel["w"] = (panel.groupby("date")["mom_7"].rank(pct=True) > 0.8).astype(float)
pnl = (panel["w"] * panel["ret"]).groupby(panel["date"]).sum()
print(pnl.mean())

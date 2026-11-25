# expect: none
import pandas as pd


def lagged_mom(s):
    return s.pct_change(14).shift(1)


panel = pd.read_parquet("crypto_panel.parquet").sort_values(["symbol", "date"])
panel["ret"] = panel.groupby("symbol")["close"].pct_change()
panel["mom"] = panel.groupby("symbol")["close"].transform(lagged_mom)
panel["w"] = panel.groupby("date")["mom"].rank(pct=True) - 0.5
pnl = (panel["w"] * panel["ret"]).groupby(panel["date"]).sum()
print(pnl.mean())

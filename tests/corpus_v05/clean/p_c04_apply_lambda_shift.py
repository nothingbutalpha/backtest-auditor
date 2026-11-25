# expect: none
import pandas as pd

panel = pd.read_parquet("crypto_panel.parquet").sort_values(["symbol", "date"])
panel["ret"] = panel.groupby("symbol")["close"].pct_change()
panel["z"] = panel.groupby("symbol", group_keys=False)["close"].apply(
    lambda s: ((s - s.rolling(30).mean()) / s.rolling(30).std()).shift(1))
panel["position"] = -panel["z"].clip(-2, 2) / 2
pnl = (panel["position"] * panel["ret"]).groupby(panel["date"]).sum()
print(pnl.mean())

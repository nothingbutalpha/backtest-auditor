# expect: LA09:medium@5
import pandas as pd

panel = pd.read_parquet("borrow_fee_panel.parquet").sort_values(["ticker", "date"])
panel["fee_pct"] = panel.groupby("ticker")["borrow_fee"].rank(pct=True)
panel["short_signal"] = (panel["fee_pct"] > 0.95).astype(int)
panel["next_ret"] = panel.groupby("ticker")["close"].pct_change().shift(-1)
print((-panel["short_signal"] * panel["next_ret"]).groupby(panel["date"]).mean().mean())

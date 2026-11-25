# expect: none
import pandas as pd
panel = pd.read_parquet("panel.parquet").sort_values(["date", "permno"])
panel["val_pct"] = panel.groupby("permno")["book_to_market"].transform(lambda s: s.expanding().rank(pct=True))
panel["signal"] = (panel["val_pct"] > 0.8).astype(int)
panel["pnl"] = panel["signal"] * panel.groupby("permno")["ret"].shift(-1)
print(panel.groupby("date")["pnl"].mean().sum())

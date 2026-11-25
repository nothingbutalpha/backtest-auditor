# expect: none
import pandas as pd

panel = pd.read_parquet("crypto_panel.parquet").sort_values(["symbol", "date"])
panel["ret"] = panel.groupby("symbol")["close"].pct_change()
panel["mom_7"] = panel.groupby("symbol")["close"].transform(lambda s: s.pct_change(7).shift(1))
panel["rank"] = panel.groupby("date")["mom_7"].rank(pct=True)
panel["w"] = (panel["rank"] > 0.8).astype(float)
daily = panel.groupby("date").apply(lambda g: (g["w"] * g["ret"]).sum() / max(g["w"].sum(), 1))
print(daily.mean() * 365)

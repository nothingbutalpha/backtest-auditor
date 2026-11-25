# expect: none
import pandas as pd
panel = pd.read_parquet("panel_long.parquet")
g = panel.groupby("ticker")["ret"]
panel["z"] = (panel["ret"] - g.transform(lambda s: s.rolling(60).mean())) / g.transform(lambda s: s.rolling(60).std())
panel["w"] = -panel.groupby("ticker")["z"].shift(1)
print((panel["w"] * panel["ret"]).groupby(panel["date"]).sum().cumsum().iloc[-1])

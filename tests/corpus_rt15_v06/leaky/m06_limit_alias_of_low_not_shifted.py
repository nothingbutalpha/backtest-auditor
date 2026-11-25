# leak: 6
import pandas as pd
bars = pd.read_csv("es_5m.csv")
bar_low = bars["low"].copy()
limit_px = bars["close"] * 0.998
hit = bar_low <= limit_px
pnl = (bars["close"].shift(-1) / limit_px - 1).where(hit, 0.0)
print(pnl.sum())

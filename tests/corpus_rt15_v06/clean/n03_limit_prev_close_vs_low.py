# expect: none
import pandas as pd
bars = pd.read_csv("es_5m.csv")
limit_px = bars["close"].shift(1) * 0.998
hit = bars["low"] <= limit_px
pnl = (bars["close"] / limit_px - 1).where(hit, 0.0)
print(pnl.sum())

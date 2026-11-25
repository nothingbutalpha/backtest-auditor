# expect: none
import pandas as pd
bars = pd.read_csv("es_5m.csv")
limit_px = bars["close"] * 0.998
next_low = bars["low"].shift(-1)
next_close = bars["close"].shift(-1)
hit = next_low <= limit_px
pnl = (next_close / limit_px - 1).where(hit, 0.0)
print(pnl.sum())

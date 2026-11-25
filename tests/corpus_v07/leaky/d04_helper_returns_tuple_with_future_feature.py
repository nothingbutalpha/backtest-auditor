# expect: LA01|LA10@7
import numpy as np
import pandas as pd

def build(px):
    trend = px.rolling(50).mean()
    peek = px.shift(-3).rolling(3).max()
    return trend, peek

px = pd.read_csv("btc_daily.csv", index_col=0, parse_dates=True)["close"]
trend, peek = build(px)
long_ = (px > trend) & (peek > px)
pnl = long_.astype(float).shift(1) * px.pct_change()
print(pnl.sum())

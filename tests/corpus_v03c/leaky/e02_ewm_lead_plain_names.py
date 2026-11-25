# expect: LA01|LA10:high
# An ewm chain shifted forward 5 bars, then a z-score, then a lagged position (all plain variable names).
import numpy as np
import pandas as pd

s = pd.read_csv("px.csv", index_col=0, parse_dates=True)["close"]
r = s.pct_change()
x = s.ewm(span=10).mean().shift(-5)
z = (x - s) / s.rolling(50).std()
position = np.sign(z).shift(1)
pnl = position * r - 0.0005 * position.diff().abs()

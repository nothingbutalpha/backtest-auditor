# expect: none
# Twin of leaky/e02: the forward-shifted ewm is only used to evaluate a lagged position.
import numpy as np
import pandas as pd

s = pd.read_csv("px.csv", index_col=0, parse_dates=True)["close"]
r = s.pct_change()
future_trend = s.ewm(span=10).mean().shift(-5) / s - 1
z = (s - s.rolling(50).mean()) / s.rolling(50).std()
position = np.sign(z).shift(1)
print("IC vs future trend", position.corr(future_trend))
pnl = position * r - 0.0005 * position.diff().abs()

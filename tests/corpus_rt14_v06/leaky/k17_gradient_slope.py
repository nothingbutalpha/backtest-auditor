# leak: 6
import numpy as np
import pandas as pd
s = pd.read_csv("eth.csv")["close"]
ema = s.ewm(span=10).mean()
slope = np.gradient(ema.to_numpy())
pos = (slope > 0).astype(float)
strat = pd.Series(pos).shift(1) * s.pct_change()
print(strat.sum())

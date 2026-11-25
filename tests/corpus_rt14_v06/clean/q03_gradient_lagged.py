# expect: none
import numpy as np
import pandas as pd
s = pd.read_csv("eth.csv")["close"]
ema = s.ewm(span=10).mean()
slope = pd.Series(np.gradient(ema.to_numpy())).shift(2)
pos = (slope > 0).astype(float)
strat = pos.shift(1) * s.pct_change()
print(strat.sum())

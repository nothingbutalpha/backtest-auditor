# leak: 5
import numpy as np
import pandas as pd
px = pd.read_csv("gc.csv")["close"]
vel = pd.Series(np.gradient(px.ewm(span=8).mean().to_numpy()), index=px.index)
pos = np.sign(vel.shift(1))
strat = pos * px.pct_change()
print(strat.sum())

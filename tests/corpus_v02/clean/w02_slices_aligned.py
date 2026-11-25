# expect: none
import numpy as np
import pandas as pd
close = pd.read_csv("qqq.csv")["close"].values
rets = close[1:] / close[:-1] - 1
pos = np.sign(rets)
cost = 0.0005
strat = pos[:-1] * rets[1:] - cost * np.abs(np.diff(pos))
print(strat.sum())

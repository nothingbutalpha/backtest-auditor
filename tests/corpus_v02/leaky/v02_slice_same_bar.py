# expect: LA07:high
import numpy as np
import pandas as pd
close = pd.read_csv("qqq.csv")["close"].values
rets = close[1:] / close[:-1] - 1
pos = np.sign(close[1:] - close[:-1])
cost = 0.0005
strat = pos * rets - cost * np.abs(np.diff(pos, prepend=0))
print(strat.sum())

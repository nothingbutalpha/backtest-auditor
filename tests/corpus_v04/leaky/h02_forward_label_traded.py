# expect: LA13:high
import numpy as np
import pandas as pd

c = pd.read_csv("btc.csv")["close"].to_numpy()
labels = np.zeros(len(c))
for i in range(len(c) - 21):
    if c[i + 1: i + 21].max() >= c[i] * 1.05:
        labels[i] = 1
ret = np.r_[np.diff(c) / c[:-1], 0.0]
pnl = labels[:-1] * ret[1:] - 0.001
print(pnl.sum())

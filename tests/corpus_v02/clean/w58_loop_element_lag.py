# expect: none
import numpy as np
import pandas as pd
c = pd.read_csv('spy.csv')['close'].to_numpy()
r = np.zeros_like(c)
r[1:] = c[1:] / c[:-1] - 1
ma = pd.Series(c).rolling(50).mean().to_numpy()
pos = np.zeros_like(c)
for i in range(len(c) - 1):
    pos[i + 1] = 1.0 if c[i] > ma[i] else 0.0
pnl = pos * r - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.sum())

# expect: none
LAG = 2
import numpy as np
import pandas as pd
c = pd.read_csv('spy.csv')['close'].to_numpy()
r = c[1:] / c[:-1] - 1
sig = np.sign(r)
pos = np.zeros_like(r)
pos[LAG:] = sig[:-LAG]
pnl = pos * r - 0.001 * np.abs(np.diff(pos, prepend=0))
print(pnl.sum())

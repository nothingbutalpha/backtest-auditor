# expect: LA07:high
import numpy as np
import pandas as pd
c = pd.read_csv('spy.csv')['close'].to_numpy()
ret = np.zeros_like(c)
ret[1:] = c[1:] / c[:-1] - 1
ma = pd.Series(c).rolling(20).mean().to_numpy()
sig = c > ma
pos = np.zeros_like(ret)
pos[1:] = np.where(sig[1:], 1, -1)
pnl = pos * ret - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.mean() / pnl.std() * np.sqrt(252))

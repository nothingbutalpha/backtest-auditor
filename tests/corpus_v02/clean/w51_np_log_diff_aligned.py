# expect: none
import numpy as np
import pandas as pd
c = pd.read_csv("spy.csv")["close"].to_numpy()
rets = np.diff(np.log(c))
pos = np.sign(rets)
pnl = pos[:-1] * rets[1:] - 0.0005 * np.abs(np.diff(pos))
print(pnl.sum())

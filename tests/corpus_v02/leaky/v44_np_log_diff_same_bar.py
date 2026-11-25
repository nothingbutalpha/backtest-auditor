# expect: LA07:high
import numpy as np
import pandas as pd
c = pd.read_csv("spy.csv")["close"].to_numpy()
rets = np.diff(np.log(c))
pos = np.sign(rets)
pnl = pos * rets - 0.0005 * np.abs(np.diff(pos, prepend=0))
print(pnl.sum())

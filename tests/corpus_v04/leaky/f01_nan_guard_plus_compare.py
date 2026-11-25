# expect: LA13:high
import numpy as np
import pandas as pd

px = pd.read_csv("spy.csv")["close"].to_numpy()
sig = np.zeros(len(px))
for i in range(len(px) - 1):
    if not np.isnan(px[i + 1]) and px[i + 1] > px[i]:
        sig[i] = 1
ret = np.r_[np.diff(px) / px[:-1], 0.0]
print((sig * ret).sum())

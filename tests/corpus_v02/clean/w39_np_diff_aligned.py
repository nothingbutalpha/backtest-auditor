# expect: none
import numpy as np
import pandas as pd
c = pd.read_csv("spy.csv")["Close"].to_numpy()
r = np.diff(c) / c[:-1]
ma = pd.Series(c).rolling(20).mean().to_numpy()
sig = (c[:-1] > ma[:-1]).astype(float)
net = sig * r - 0.001 * np.abs(np.diff(sig, prepend=0))
print(net.mean() * 252)

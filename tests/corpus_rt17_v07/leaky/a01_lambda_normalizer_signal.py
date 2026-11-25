# leak: 5
import numpy as np
import pandas as pd
px = pd.read_csv("etf_closes.csv", index_col=0, parse_dates=True)["XLE"]
norm = lambda s: (s - s.mean()) / s.std()
mom = px.pct_change(60)
score = norm(mom)
pos = np.sign(score).shift(1)
pnl = pos * px.pct_change()
print(pnl.sum())

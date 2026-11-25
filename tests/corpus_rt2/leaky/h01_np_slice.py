import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
c = df["Close"].to_numpy()
r = c[1:] / c[:-1] - 1
sig = np.where(c[1:] > c[:-1], 1, -1)
strat = sig * r
print(strat.mean() / strat.std() * np.sqrt(252))

import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
c = df["Close"].to_numpy()
r = c[1:] / c[:-1] - 1
sig = np.sign(r[:-1])
strat = sig * r[1:] - 0.0005 * np.abs(np.diff(sig, prepend=0))
print(strat.mean())

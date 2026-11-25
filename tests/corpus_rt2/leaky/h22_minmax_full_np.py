import numpy as np, pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
x = df["Close"].pct_change(10).to_numpy()
xn = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
df["position"] = np.where(xn > 0.5, 1, 0)
df["strat"] = df["position"].shift(1) * df["Close"].pct_change()

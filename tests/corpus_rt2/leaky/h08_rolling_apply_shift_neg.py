import pandas as pd, numpy as np
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
ret = df["Close"].pct_change()
df["vol21"] = ret.rolling(21).apply(lambda w: np.std(w)).shift(-21)
df["weight"] = 0.1 / df["vol21"]
df["strat"] = df["weight"].shift(1) * ret

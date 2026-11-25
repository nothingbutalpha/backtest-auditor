# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["signal"] = np.sign(df["close"].pct_change(60))
df["strat"] = np.copysign(df["fwd_ret"], df["signal"])
print(df["strat"].mean() * 252)

# expect: LA10:high
import numpy as np
import pandas as pd

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["strat"] = np.copysign(df["fwd_ret"], df["fwd_ret"].rolling(3).mean())
print(df["strat"].mean() * 252)

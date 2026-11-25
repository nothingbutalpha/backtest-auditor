# expect: LA10:high
import numpy as np
import pandas as pd

df = pd.read_csv("qqq.csv", index_col="date", parse_dates=True)
df["next_ret"] = df["close"].pct_change().shift(-1)
df["fwd_20d"] = df["close"].shift(-20) / df["close"] - 1
pnl = np.copysign(df["next_ret"], df["fwd_20d"].rank(pct=True) - 0.5)
print(pnl.mean() * 252)

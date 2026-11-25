# expect: none
import numpy as np
import pandas as pd

df = pd.read_csv("tlt.csv", index_col="date", parse_dates=True)
df["ret"] = df["close"].pct_change()
pnl = np.copysign(df["ret"], df["close"].pct_change(120).shift(1)) - 0.0002
print(pnl.mean() * 252)

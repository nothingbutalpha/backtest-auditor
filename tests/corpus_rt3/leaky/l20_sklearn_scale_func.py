# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom_z"] = scale(df["mom"])
df["signal"] = (df["mom_z"] > 1).astype(int)

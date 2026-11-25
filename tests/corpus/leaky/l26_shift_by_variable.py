# expect: LA01:high, LA10:high, CO01:possible
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_csv("spy.csv")
h = -1
df["feature"] = df["close"].pct_change().shift(h)
df["signal"] = (df["feature"] > 0).astype(int)
df["strategy"] = df["signal"].shift(1) * df["close"].pct_change()

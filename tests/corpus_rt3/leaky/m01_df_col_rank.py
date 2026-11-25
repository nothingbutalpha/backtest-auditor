# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("vix.csv", index_col=0, parse_dates=True)
df["signal"] = df["close"].rank(pct=True)
df["position"] = (df["signal"] < 0.2).astype(int).shift(1)

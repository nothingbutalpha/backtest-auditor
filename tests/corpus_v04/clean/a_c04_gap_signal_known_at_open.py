# expect: none
"""Gap-down fade: the signal uses today's open and yesterday's close, both known at the open."""
import pandas as pd

df = pd.read_csv("aapl.csv", index_col=0, parse_dates=True)
gap = df["open"] / df["close"].shift(1) - 1
entries = gap < -0.03
entry_px = df["open"].where(entries)
exit_px = df["close"].where(entries)
day_ret = (exit_px / entry_px - 1 - 0.001).dropna()
print(day_ret.mean(), len(day_ret))

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import vectorbt as vbt
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
fast = df["close"].rolling(10).mean()
slow = df["close"].rolling(50).mean()
entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).shift(1, fill_value=False)
exits = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).shift(1, fill_value=False)
pf = vbt.Portfolio.from_signals(df["close"], entries, exits, price=df["open"], fees=0.0005)
print(pf.stats())

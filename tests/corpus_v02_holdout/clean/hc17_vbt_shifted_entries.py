# expect: none
# Looks like a vectorbt same-close fill, but entries/exits are shifted one bar and fees/slippage are set.
import pandas as pd
import vectorbt as vbt

df = pd.read_csv("eth_daily.csv", index_col="date", parse_dates=True)
close = df["close"]
fast = close.rolling(10).mean()
slow = close.rolling(30).mean()
entries = ((fast > slow) & (fast.shift(1) <= slow.shift(1))).shift(1, fill_value=False)
exits = ((fast < slow) & (fast.shift(1) >= slow.shift(1))).shift(1, fill_value=False)

pf = vbt.Portfolio.from_signals(close, entries, exits, price=df["open"], fees=0.001, slippage=0.0005, freq="1D")
print(pf.stats())

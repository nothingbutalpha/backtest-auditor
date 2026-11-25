# expect: LA07:possible, CO01:possible
import pandas as pd
import vectorbt as vbt

close = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")["close"]
fast_ma = close.rolling(10).mean()
slow_ma = close.rolling(50).mean()
entries = fast_ma > slow_ma
exits = fast_ma < slow_ma
pf = vbt.Portfolio.from_signals(close, entries, exits)
print(pf.stats())

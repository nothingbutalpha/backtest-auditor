# expect: none
import pandas as pd
import vectorbt as vbt

close = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")["close"]
fast_ma = close.rolling(10).mean()
slow_ma = close.rolling(50).mean()
entries = fast_ma > slow_ma
exits = fast_ma < slow_ma
pf = vbt.Portfolio.from_signals(close, entries.shift(1, fill_value=False), exits.shift(1, fill_value=False),
                                fees=0.001, slippage=0.0005)

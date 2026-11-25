# expect: none
import pandas as pd
import vectorbt as vbt
close = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
fast, slow = close.rolling(10).mean(), close.rolling(50).mean()
entries = (fast > slow).shift(1, fill_value=False)
exits = (fast < slow).shift(1, fill_value=False)
pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.0005, slippage=0.0005)
print(pf.sharpe_ratio())

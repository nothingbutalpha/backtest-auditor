# expect: LA07:medium
import vectorbt as vbt
import pandas as pd
close = pd.read_csv("btc.csv", index_col=0, parse_dates=True)["close"]
rsi = vbt.RSI.run(close, window=14).rsi
entries = rsi < 30
exits = rsi > 70
pf = vbt.Portfolio.from_signals(close, entries, exits, price=close, fees=0.001, slippage=0.0005)
print(pf.sharpe_ratio())

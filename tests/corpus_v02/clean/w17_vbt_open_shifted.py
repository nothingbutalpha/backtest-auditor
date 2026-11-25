# expect: none
import vectorbt as vbt
import pandas as pd
data = pd.read_csv("btc.csv", index_col=0, parse_dates=True)
close, open_ = data["close"], data["open"]
rsi = vbt.RSI.run(close, window=14).rsi
entries = (rsi < 30).shift(1, fill_value=False)
exits = (rsi > 70).shift(1, fill_value=False)
pf = vbt.Portfolio.from_signals(close, entries, exits, price=open_, fees=0.001, slippage=0.0005)
print(pf.sharpe_ratio())

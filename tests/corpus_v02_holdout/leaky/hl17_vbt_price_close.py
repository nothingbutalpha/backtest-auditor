# expect: LA07
# vectorbt fills at the same close that produced the unshifted crossover entries.
import pandas as pd
import vectorbt as vbt

close = pd.read_csv("eth_daily.csv", index_col="date", parse_dates=True)["close"]
fast = close.rolling(10).mean()
slow = close.rolling(30).mean()
entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))

pf = vbt.Portfolio.from_signals(close, entries, exits, price=close, fees=0.001, freq="1D")
print(pf.stats())
print("Sharpe:", pf.sharpe_ratio())

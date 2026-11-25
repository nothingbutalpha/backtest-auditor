# leak: 9
import pandas as pd
import vectorbt as vbt
bars = pd.read_parquet("qqq.parquet")
fast = bars["close"].ewm(span=12).mean()
slow = bars["close"].ewm(span=26).mean()
buy = fast > slow
sell = fast < slow
pf = vbt.Portfolio.from_signals(close=bars["open"], entries=buy, exits=sell)
print(pf.total_return())

# expect: none
import pandas as pd
import vectorbt as vbt
bars = pd.read_parquet("qqq.parquet")
fast = bars["close"].ewm(span=12).mean()
slow = bars["close"].ewm(span=26).mean()
buy = (fast > slow).shift(1, fill_value=False)
sell = (fast < slow).shift(1, fill_value=False)
pf = vbt.Portfolio.from_signals(bars["close"], entries=buy, exits=sell, price=bars["open"])
print(pf.total_return())

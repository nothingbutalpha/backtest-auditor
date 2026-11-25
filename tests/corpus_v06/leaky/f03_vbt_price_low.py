# expect: LA07:medium@10
import pandas as pd
import vectorbt as vbt

data = pd.read_parquet("bnb_1h.parquet")
close, low = data["close"], data["low"]
entries = close < close.rolling(24).mean() * 0.98
exits = close > close.rolling(24).mean()
pf = vbt.Portfolio.from_signals(close, entries.shift(1, fill_value=False), exits.shift(1, fill_value=False),
                                price=low, fees=0.001)
print(pf.total_return())

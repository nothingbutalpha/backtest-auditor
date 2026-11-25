import vectorbt as vbt
data = vbt.YFData.download("SPY")
close = data.get("Close")
open_ = data.get("Open")
entries = close > close.rolling(50).mean()
exits = close < close.rolling(50).mean()
pf = vbt.Portfolio.from_signals(close, entries.shift(1), exits.shift(1), price=open_, fees=0.001)

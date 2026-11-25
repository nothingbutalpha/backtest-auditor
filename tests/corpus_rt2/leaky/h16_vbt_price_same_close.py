import vectorbt as vbt
price = vbt.YFData.download("SPY").get("Close")
fast = vbt.MA.run(price, 10)
slow = vbt.MA.run(price, 50)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)
pf = vbt.Portfolio.from_signals(price, entries, exits, price=price, fees=0.001)
print(pf.total_return())

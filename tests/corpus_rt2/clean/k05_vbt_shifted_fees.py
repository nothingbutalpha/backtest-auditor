import vectorbt as vbt
price = vbt.YFData.download("SPY").get("Close")
fast = vbt.MA.run(price, 10)
slow = vbt.MA.run(price, 50)
entries = fast.ma_crossed_above(slow).shift(1)
exits = fast.ma_crossed_below(slow).shift(1)
pf = vbt.Portfolio.from_signals(price, entries, exits, fees=0.001, slippage=0.0005)
print(pf.sharpe_ratio())

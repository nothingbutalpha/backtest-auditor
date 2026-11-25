import backtrader as bt
cerebro = bt.Cerebro(cheat_on_close=True)
cerebro.broker.setcommission(commission=0.001)

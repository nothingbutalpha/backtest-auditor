# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import bt
s = bt.Strategy("sma", [bt.algos.SelectWhere(signal.shift(1) > 0), bt.algos.WeighEqually(), bt.algos.Rebalance()])
test = bt.Backtest(s, data, commissions=lambda q, p: abs(q) * p * 0.001)
res = bt.run(test)
res.display()

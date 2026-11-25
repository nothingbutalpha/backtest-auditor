# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import vectorbt as vbt
cfg = dict(fees=0.001, slippage=0.0005)
pf = vbt.Portfolio.from_signals(close, entries.shift(1, fill_value=False), exits.shift(1, fill_value=False), **cfg)
print(pf.sharpe_ratio())

# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
wide = pd.read_csv("closes.csv", index_col=0, parse_dates=True).sort_index()
wide = wide.sort_index(axis=1, ascending=False)
ret = wide.pct_change()
ma = wide.rolling(100).mean()
signal = (wide > ma).astype(float)
pos = signal.shift(1)
strat = (pos * ret).mean(axis=1) - 0.0005 * pos.diff().abs().mean(axis=1)

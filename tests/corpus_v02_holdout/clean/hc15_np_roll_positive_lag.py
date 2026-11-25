# expect: none
# Looks like an np.roll offset leak, but np.roll(x, 1) lags by one bar and the wrapped value is zeroed.
import numpy as np
import pandas as pd

close = pd.read_csv("spy_daily.csv")["close"].to_numpy(dtype=float)
ret = np.zeros_like(close)
ret[1:] = close[1:] / close[:-1] - 1

yesterday = np.roll(close, 1)
yesterday[0] = np.nan
signal = np.where(close > yesterday, 1.0, 0.0)
position = np.roll(signal, 1)
position[0] = 0.0

strat = position * ret - 0.0005 * np.abs(np.diff(position, prepend=0.0))
print(f"Sharpe: {strat.mean() / strat.std() * np.sqrt(252):.2f}")

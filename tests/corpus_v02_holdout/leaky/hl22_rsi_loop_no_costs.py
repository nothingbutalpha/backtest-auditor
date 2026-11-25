# expect: CO01:possible
# Bar-by-bar RSI strategy compounds on prior-bar positions and reports CAGR with no costs.
import numpy as np
import pandas as pd

close = pd.read_csv("ndx.csv")["close"].to_numpy()
delta = np.diff(close, prepend=close[0])
gain = pd.Series(np.clip(delta, 0, None)).rolling(14).mean().to_numpy()
loss = pd.Series(np.clip(-delta, 0, None)).rolling(14).mean().to_numpy()
rsi = 100 - 100 / (1 + gain / loss)

equity = [1.0]
pos = 0
for i in range(15, len(close)):
    equity.append(equity[-1] * (1 + pos * (close[i] / close[i - 1] - 1)))
    if rsi[i] < 30:
        pos = 1
    elif rsi[i] > 70:
        pos = 0

years = len(equity) / 252
print("CAGR:", equity[-1] ** (1 / years) - 1)

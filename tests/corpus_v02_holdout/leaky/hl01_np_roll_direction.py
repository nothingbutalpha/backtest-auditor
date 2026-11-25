# expect: LA01
# Direction signal compares today's close with np.roll(close, -1), which is tomorrow's close.
import numpy as np
import pandas as pd

df = pd.read_csv("spy_daily.csv", parse_dates=["date"]).sort_values("date")
close = df["close"].to_numpy(dtype=float)

ret = np.zeros_like(close)
ret[1:] = close[1:] / close[:-1] - 1

tomorrow = np.roll(close, -1)
signal = np.where(tomorrow > close, 1.0, 0.0)
position = np.roll(signal, 1)
position[0] = 0.0

cost = 0.0005 * np.abs(np.diff(position, prepend=0.0))
strat = position * ret - cost
sharpe = strat.mean() / strat.std() * np.sqrt(252)
print(f"Sharpe: {sharpe:.2f}")

# expect: LA11:medium
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
ret = df["close"].pct_change()
best_sharpe, best_params = -np.inf, None
for fast in range(5, 50, 5):
    for slow in range(20, 200, 10):
        pos = (df["close"].rolling(fast).mean() > df["close"].rolling(slow).mean()).astype(int).shift(1)
        strat = pos * ret - pos.diff().abs() * 0.0005
        sharpe = strat.mean() / strat.std() * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe, best_params = sharpe, (fast, slow)
print("best", best_params, best_sharpe)

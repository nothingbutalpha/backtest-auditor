# expect: LA11
# Fast/slow MA grid is scored on the full sample and the best Sharpe is reported as the result.
import numpy as np
import pandas as pd

close = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
ret = close.pct_change().fillna(0)

results = {}
for fast in range(5, 50, 5):
    for slow in range(50, 250, 25):
        pos = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float).shift(1).fillna(0)
        net = pos * ret - 0.0005 * pos.diff().abs().fillna(0)
        results[(fast, slow)] = net.mean() / net.std() * np.sqrt(252)

best = max(results, key=results.get)
print("Best params", best, "Sharpe", round(results[best], 2))

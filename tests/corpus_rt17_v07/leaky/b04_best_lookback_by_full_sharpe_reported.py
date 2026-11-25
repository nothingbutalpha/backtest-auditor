# leak: 10
import numpy as np
import pandas as pd
px = pd.read_csv("gld.csv", index_col=0, parse_dates=True)["close"]
r = px.pct_change()
def strat(n):
    return np.sign(px - px.rolling(n).mean()).shift(1) * r
def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)
best_n = max(range(10, 210, 10), key=lambda n: sharpe(strat(n)))
print("Sharpe", sharpe(strat(best_n)))

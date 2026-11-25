# expect: none
# Looks like an in-sample parameter grid, but params are picked on 2010-2018 and evaluated on 2019+.
import numpy as np
import pandas as pd

close = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]


def net_returns(px, fast, slow, fee=0.0005):
    pos = (px.rolling(fast).mean() > px.rolling(slow).mean()).astype(float).shift(1).fillna(0)
    return pos * px.pct_change().fillna(0) - fee * pos.diff().abs().fillna(0)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


train_px = close.loc["2010":"2018"]
scores = {(f, s): sharpe(net_returns(train_px, f, s)) for f in range(5, 50, 5) for s in range(50, 250, 25)}
fast, slow = max(scores, key=scores.get)

oos = net_returns(close.loc["2018-01-01":], fast, slow).loc["2019":]
print(f"Chosen on train: {fast}/{slow}; out-of-sample Sharpe {sharpe(oos):.2f}")

# expect: none
# Parameters are chosen on each training window and evaluated on the next, untouched window.
import numpy as np
import pandas as pd

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
ret = df["close"].pct_change()
cost = 0.0005


def backtest(close, window):
    pos = (close > close.rolling(window).mean()).astype(int).shift(1)
    return pos * close.pct_change() - pos.diff().abs() * cost


oos = []
years = sorted(set(df.index.year))
for test_year in years[3:]:
    train = df[(df.index.year >= test_year - 3) & (df.index.year < test_year)]
    test = df[df.index.year == test_year]
    best_w, best_sharpe = None, -np.inf
    for w in [20, 50, 100, 200]:
        r_train = backtest(train["close"], w)
        s = r_train.mean() / r_train.std()
        if s > best_sharpe:
            best_w, best_sharpe = w, s
    oos.append(backtest(test["close"], best_w))

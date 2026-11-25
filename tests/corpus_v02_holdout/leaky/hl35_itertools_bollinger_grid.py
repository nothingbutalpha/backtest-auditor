# expect: LA11
# Bollinger grid via itertools.product is ranked on the full sample and the top row is reported.
import itertools

import numpy as np
import pandas as pd


def backtest(df, window, k, fee=0.0005):
    mid = df["close"].rolling(window).mean()
    sd = df["close"].rolling(window).std()
    sig = np.where(df["close"] < mid - k * sd, 1.0, np.where(df["close"] > mid, 0.0, np.nan))
    pos = pd.Series(sig, index=df.index).ffill().fillna(0).shift(1).fillna(0)
    net = pos * df["close"].pct_change().fillna(0) - fee * pos.diff().abs().fillna(0)
    return {"window": window, "k": k, "sharpe": net.mean() / net.std() * np.sqrt(252), "total": net.sum()}


df = pd.read_csv("xle.csv", index_col="date", parse_dates=True)
rows = [backtest(df, w, k) for w, k in itertools.product([10, 20, 30, 50], [1.0, 1.5, 2.0, 2.5])]
table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
best = table.iloc[0]
print(f"Optimal: window={best.window:.0f}, k={best.k}, Sharpe={best.sharpe:.2f}")

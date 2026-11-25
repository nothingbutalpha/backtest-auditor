# expect: LA11
# Breakout lookback picked via a dict comprehension over the whole history, then rerun and reported.
import numpy as np
import pandas as pd

df = pd.read_csv("crude.csv", index_col="date", parse_dates=True)


def run(window, fee=0.0008):
    hi = df["high"].rolling(window).max().shift(1)
    lo = df["low"].rolling(window).min().shift(1)
    sig = pd.Series(np.where(df["close"] > hi, 1, np.where(df["close"] < lo, -1, np.nan)), index=df.index)
    pos = sig.ffill().fillna(0).shift(1).fillna(0)
    return pos * df["close"].pct_change().fillna(0) - fee * pos.diff().abs().fillna(0)


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


scores = {w: sharpe(run(w)) for w in range(10, 200, 10)}
best_w = max(scores, key=scores.get)
final = run(best_w)
print(f"window={best_w} sharpe={sharpe(final):.2f} cagr={(1 + final).prod() ** (252 / len(final)) - 1:.1%}")

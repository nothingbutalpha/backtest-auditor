# expect: none
# Looks like a same-interval slice product, but sig[:-1] is paired with the following return r[1:].
import numpy as np

c = np.loadtxt("gold_close.csv", delimiter=",", skiprows=1, usecols=1)
r = c[1:] / c[:-1] - 1
sig = np.where(c[1:] > c[:-1] * 1.002, 1.0, 0.0)

held = sig[:-1]
turnover = np.abs(np.diff(held, prepend=0.0))
strat = held * r[1:] - 0.0003 * turnover
equity = np.cumprod(1 + strat)
print("final", equity[-1], "sharpe", strat.mean() / strat.std() * np.sqrt(252))

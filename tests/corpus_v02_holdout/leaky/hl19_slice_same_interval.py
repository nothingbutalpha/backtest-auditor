# expect: LA07
# Signal from c[1:] vs c[:-1] is multiplied by the return over that same c[:-1] -> c[1:] interval.
import numpy as np

c = np.loadtxt("gold_close.csv", delimiter=",", skiprows=1, usecols=1)
r = c[1:] / c[:-1] - 1
sig = np.where(c[1:] > c[:-1] * 1.002, 1.0, 0.0)

turnover = np.abs(np.diff(sig, prepend=0.0))
strat = sig * r - 0.0003 * turnover
equity = np.cumprod(1 + strat)
print("final", equity[-1], "sharpe", strat.mean() / strat.std() * np.sqrt(252))

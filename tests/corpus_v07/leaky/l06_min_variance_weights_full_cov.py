# expect: LA05|LA09@6
import numpy as np
import pandas as pd

r = pd.read_csv("factor_etfs.csv", index_col=0, parse_dates=True).pct_change().dropna()
sigma = r.cov().values
ones = np.ones(sigma.shape[0])
w = np.linalg.solve(sigma, ones)
w = w / w.sum()
port = r @ w
print("vol", port.std() * 252 ** 0.5, "Sharpe", port.mean() / port.std() * 252 ** 0.5)

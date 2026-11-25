# expect: none
import numpy as np
import pandas as pd

r = pd.read_csv("factor_etfs.csv", index_col=0, parse_dates=True).pct_change().dropna()
out = {}
for i in range(252, len(r)):
    sigma = r.iloc[i - 252:i].cov().values
    w = np.linalg.solve(sigma, np.ones(len(sigma)))
    out[r.index[i]] = r.iloc[i].values @ (w / w.sum())
port = pd.Series(out)
print(port.std() * 252 ** 0.5)

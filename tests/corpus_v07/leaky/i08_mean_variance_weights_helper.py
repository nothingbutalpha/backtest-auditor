# expect: LA05|LA09@6
import numpy as np
import pandas as pd

def tangency(r):
    mu = r.mean().values
    inv = np.linalg.pinv(r.cov().values)
    w = inv @ mu
    return w / w.sum()

rets = pd.read_parquet("country_etf_monthly.parquet")
w = tangency(rets)
book = rets @ w
print("Sharpe", book.mean() / book.std() * 12 ** 0.5)

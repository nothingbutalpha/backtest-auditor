# expect: none
import numpy as np
import pandas as pd

def tangency(r):
    mu = r.mean().values
    inv = np.linalg.pinv(r.cov().values)
    w = inv @ mu
    return w / w.sum()

rets = pd.read_parquet("country_etf_monthly.parquet")
w = tangency(rets.loc[:"2015"])
book = rets.loc["2016":] @ w
print("OOS Sharpe", book.mean() / book.std() * 12 ** 0.5)

# expect: LA05|LA11@12
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

px = np.log(pd.read_parquet("bank_close.parquet"))
book = []
for a in px.columns:
    for b in px.columns:
        if a < b:
            spr = px[a] - px[b]
            if adfuller(spr)[1] < 0.01:
                z = (spr - spr.rolling(40).mean()) / spr.rolling(40).std()
                book.append(-np.sign(z).shift(1) * spr.diff())
pnl = pd.concat(book, axis=1).mean(axis=1)
print(pnl.mean() / pnl.std() * 252 ** 0.5)

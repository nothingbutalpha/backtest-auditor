# leak: 6
import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint as eg_test
closes = pd.read_parquet("banks_close.parquet")
pairs = [(u, v) for u, v in itertools.combinations(closes.columns, 2) if eg_test(closes[u], closes[v])[1] < 0.05]
book = []
for u, v in pairs:
    s = closes[u] - closes[v]
    z = (s - s.rolling(60).mean()) / s.rolling(60).std()
    book.append((-z.clip(-2, 2)).shift(1) * s.diff())
print(pd.concat(book, axis=1).sum(axis=1).cumsum().iloc[-1])

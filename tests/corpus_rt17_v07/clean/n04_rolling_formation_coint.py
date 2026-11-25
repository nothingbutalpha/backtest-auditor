# expect: none
import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint
closes = pd.read_parquet("banks_close.parquet")
legs = []
for t in range(252, len(closes) - 63, 63):
    look = closes.iloc[t - 252:t]
    trade = closes.iloc[t:t + 63]
    for u, v in itertools.combinations(closes.columns, 2):
        if coint(look[u], look[v])[1] < 0.05:
            s = trade[u] - trade[v]
            mu, sd = (look[u] - look[v]).mean(), (look[u] - look[v]).std()
            legs.append((-((s - mu) / sd).clip(-2, 2)).shift(1) * s.diff())
print(pd.concat(legs).sum())

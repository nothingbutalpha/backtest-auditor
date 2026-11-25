# expect: none
import numpy as np
import pandas as pd


def forward_hit_labels(c, horizon=20, target=0.05):
    out = np.zeros(len(c))
    for i in range(len(c) - horizon):
        if c[i + 1: i + 1 + horizon].max() >= c[i] * (1 + target):
            out[i] = 1
    return out


px = pd.read_csv("btc.csv", index_col="date", parse_dates=True)["close"]
hits = pd.Series(forward_hit_labels(px.to_numpy()), index=px.index)
rate_by_year = hits.groupby(hits.index.year).mean()
print(rate_by_year)
rate_by_year.plot(kind="bar")

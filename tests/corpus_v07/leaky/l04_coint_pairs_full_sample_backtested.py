# expect: LA05|LA11@9
import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint

px = pd.read_parquet("reit_close.parquet")
chosen = []
for a, b in itertools.combinations(px.columns, 2):
    stat, p, crit = coint(px[a], px[b])
    if p < 0.05:
        chosen.append((a, b))

def trade(a, b):
    spread = px[a] - px[b]
    z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    return (-z.clip(-2, 2).shift(1) * spread.diff()).sum()

print({pair: trade(*pair) for pair in chosen})

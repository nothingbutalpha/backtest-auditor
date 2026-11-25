# expect: none
import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint

px = pd.read_parquet("reit_close.parquet")
form, trade_px = px.loc[:"2018"], px.loc["2019":]
chosen = []
for a, b in itertools.combinations(px.columns, 2):
    _, p, _ = coint(form[a], form[b])
    if p < 0.05:
        chosen.append((a, b))

def trade(a, b):
    spread = trade_px[a] - trade_px[b]
    z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    return (-z.clip(-2, 2).shift(1) * spread.diff()).sum()

print({pair: trade(*pair) for pair in chosen})

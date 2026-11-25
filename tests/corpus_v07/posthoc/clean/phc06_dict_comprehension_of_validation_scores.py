# expect: none
import numpy as np
import pandas as pd

close = pd.read_csv("btcusdt_4h.csv", index_col=0, parse_dates=True)["close"]
train, valid, test = close[:"2021-12-31"], close["2022-01-01":"2022-12-31"], close["2023-01-01":]

def run(px, lb):
    z = (px - px.rolling(lb).mean()) / px.rolling(lb).std()
    r = (-np.sign(z)).shift(1) * px.pct_change()
    return r.mean() / r.std()

valid_sr = {lb: run(valid, lb) for lb in range(10, 200, 10)}
best = max(valid_sr, key=valid_sr.get)
print("test Sharpe of the chosen lookback:", run(test, best))

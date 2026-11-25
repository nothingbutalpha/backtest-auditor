# expect: none
import numpy as np
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
train, test = df.loc[:"2015"], df.loc["2016":]


def sharpe_for(data, window):
    pos = (data["close"] > data["close"].rolling(window).mean()).astype(int).shift(1)
    r = pos * data["close"].pct_change() - 0.0005 * pos.diff().abs()
    return r.mean() / r.std() * np.sqrt(252)


results = {w: sharpe_for(train, w) for w in range(10, 210, 10)}
best = max(results, key=results.get)
print(best, sharpe_for(test, best))

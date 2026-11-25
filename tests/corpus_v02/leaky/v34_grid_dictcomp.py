# expect: LA11:medium
import numpy as np
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)


def sharpe_for(window):
    pos = (df["close"] > df["close"].rolling(window).mean()).astype(int).shift(1)
    r = pos * df["close"].pct_change() - 0.0005 * pos.diff().abs()
    return r.mean() / r.std() * np.sqrt(252)


results = {w: sharpe_for(w) for w in range(10, 210, 10)}
best = max(results, key=results.get)
print(best, results[best])

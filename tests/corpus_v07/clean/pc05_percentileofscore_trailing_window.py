# expect: none
import pandas as pd
from scipy import stats

skew = pd.read_csv("cboe_skew.csv", index_col=0, parse_dates=True)["skew"]
skew_pct = skew.rolling(252).apply(lambda w: stats.percentileofscore(w[:-1], w[-1]), raw=True)
hedge_on = (skew_pct > 90).astype(int)
spx = pd.read_csv("spx.csv", index_col=0, parse_dates=True)["close"].pct_change()
print(((1 - 0.5 * hedge_on.shift(1)) * spx).sum())

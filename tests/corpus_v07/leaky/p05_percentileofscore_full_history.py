# expect: LA09@7
import pandas as pd
from scipy import stats

skew = pd.read_csv("cboe_skew.csv", index_col=0, parse_dates=True)["skew"]
allvals = skew.values
skew_pct = skew.apply(lambda v: stats.percentileofscore(allvals, v))
hedge_on = (skew_pct > 90).astype(int)
spx = pd.read_csv("spx.csv", index_col=0, parse_dates=True)["close"].pct_change()
print(((1 - 0.5 * hedge_on.shift(1)) * spx).sum())

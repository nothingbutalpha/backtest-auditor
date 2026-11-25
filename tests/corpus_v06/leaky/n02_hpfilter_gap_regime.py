# expect: LA02:high@7
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

spread = pd.read_csv("hy_spread_monthly.csv", index_col=0, parse_dates=True)["oas"]
cycle, trend = hpfilter(spread, lamb=14400)
risk_on = (cycle < 0).astype(float).shift(1)
eq = pd.read_csv("acwi_monthly.csv", index_col=0, parse_dates=True)["ret"].reindex(spread.index)
strat = risk_on * eq
print(strat.mean() * 12, strat.std() * np.sqrt(12))

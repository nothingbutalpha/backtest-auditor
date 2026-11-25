# expect: LA05:medium@8
import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_csv("ewc_ewa.csv", index_col=0, parse_dates=True)
a, b = np.log(px["EWC"]), np.log(px["EWA"])
hedge = sm.OLS(a, sm.add_constant(b)).fit().params["EWA"]
spread = a - hedge * b
z = (spread - spread.rolling(40).mean()) / spread.rolling(40).std()
position = (-np.sign(z) * (z.abs() > 1.5)).shift(1)
pnl = position * (a.diff() - hedge * b.diff())
print(pnl.sum())

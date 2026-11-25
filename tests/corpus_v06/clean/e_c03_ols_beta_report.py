# expect: none
import numpy as np
import pandas as pd
import statsmodels.api as sm

px = pd.read_csv("ewc_ewa.csv", index_col=0, parse_dates=True)
a, b = np.log(px["EWC"]).diff().dropna(), np.log(px["EWA"]).diff().dropna()
res = sm.OLS(a, sm.add_constant(b)).fit()
print(res.params.round(3), res.rsquared)

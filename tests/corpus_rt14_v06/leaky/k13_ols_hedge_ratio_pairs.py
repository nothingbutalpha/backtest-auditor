# leak: 7
import numpy as np
import pandas as pd
import statsmodels.api as sm
a = pd.read_csv("ko.csv", index_col=0)["close"]
b = pd.read_csv("pep.csv", index_col=0)["close"]
beta = sm.OLS(a, sm.add_constant(b)).fit().params.iloc[1]
spread = a - beta * b
zs = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
signal = -np.sign(zs).where(zs.abs() > 2, 0)
pnl = signal.shift(1) * (a.pct_change() - beta * b.pct_change())
print(pnl.sum())

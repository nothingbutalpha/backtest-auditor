# leak: 5
import pandas as pd
from statsmodels.tsa.seasonal import STL
gas = pd.read_csv("natgas.csv", index_col=0, parse_dates=True)["settle"]
res = STL(gas, period=52).fit()
deseason = gas - res.seasonal
z = (deseason - deseason.rolling(26).mean()) / deseason.rolling(26).std()
signal = (z < -2).astype(int)
ret = signal.shift(1) * gas.pct_change()
print(ret.sum())

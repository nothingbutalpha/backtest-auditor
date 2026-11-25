# expect: none
import numpy as np
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
r = df["close"].pct_change().dropna()
print("buy and hold Sharpe", r.mean() / r.std() * np.sqrt(252))

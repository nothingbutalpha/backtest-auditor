# expect: none
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
y = pd.read_csv("gld.csv", index_col=0)["close"]
x = pd.read_csv("gdx.csv", index_col=0)["close"]
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, observation_matrices=np.vstack([x, np.ones(len(x))]).T[:, None, :])
state_means, _ = kf.filter(y.values)
hedge = state_means[:, 0]
spread = y - hedge * x
signal = -np.sign(spread - spread.rolling(20).mean())
pnl = pd.Series(signal).shift(1) * y.pct_change()
print(pnl.sum())

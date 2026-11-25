# expect: none
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

px = pd.read_parquet("xom_cvx.parquet")
obs = np.vstack([px["CVX"].to_numpy(), np.ones(len(px))]).T[:, np.newaxis, :]
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, observation_matrices=obs, transition_covariance=1e-4 * np.eye(2))
means, _ = kf.filter(px["XOM"].to_numpy())
spread = px["XOM"] - means[:, 0] * px["CVX"] - means[:, 1]
position = -np.sign(spread / spread.rolling(20).std()).shift(1)
print((position * px["XOM"].pct_change()).sum())

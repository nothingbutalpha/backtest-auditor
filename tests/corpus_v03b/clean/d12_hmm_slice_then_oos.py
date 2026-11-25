# expect: none
# LA05 class (added after the first measurement): HMM fitted on a slice, then an `oos` frame predicted.
import pandas as pd
from hmmlearn.hmm import GaussianHMM

X = pd.read_parquet("regime_feats.parquet")
hmm = GaussianHMM(n_components=3, n_iter=100, random_state=1).fit(X.loc[:"2014"].to_numpy())
oos = X.loc["2015":]
oos_states = hmm.predict(oos.to_numpy())

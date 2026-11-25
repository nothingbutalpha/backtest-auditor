# expect: none
# LA05 class: a regime model fitted on data up to 2014, then predicted one later date at a time.
import pandas as pd
from hmmlearn.hmm import GaussianHMM

px = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
X = pd.DataFrame({"ret": px.pct_change(), "vol": px.pct_change().rolling(20).std()}).dropna()
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=0)
hmm.fit(X.loc[:"2014"].to_numpy())
states = {}
for d in X.loc["2015":].index:
    states[d] = hmm.predict(X.loc[:d].to_numpy())[-1]
regime = pd.Series(states)

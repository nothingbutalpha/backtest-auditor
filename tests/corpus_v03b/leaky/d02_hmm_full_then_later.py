# expect: LA05
# Twin of clean/d02: the regime model is fitted on all dates, then "predicted" on the later dates it has seen.
import pandas as pd
from hmmlearn.hmm import GaussianHMM

px = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
X = pd.DataFrame({"ret": px.pct_change(), "vol": px.pct_change().rolling(20).std()}).dropna()
hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=0)
hmm.fit(X.to_numpy())
regime = pd.Series(hmm.predict(X.loc["2015":].to_numpy()), index=X.loc["2015":].index)

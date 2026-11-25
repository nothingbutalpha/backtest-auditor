# expect: none
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

d = pd.read_parquet("downgrade_data.parquet").sort_values("asof")
train = d[d["asof"] < "2018-01-01"]
val = d[(d["asof"] >= "2018-01-01") & (d["asof"] < "2020-01-01")]
test = d[d["asof"] >= "2020-01-01"]
cols = ["lev", "icr", "fcf_margin"]
clf = LogisticRegression().fit(train[cols], train["downgrade"])
pv = clf.predict_proba(val[cols])[:, 1]
cut = max(np.arange(0.1, 0.9, 0.02), key=lambda t: f1_score(val["downgrade"], pv > t))
print("threshold", cut, "test F1", f1_score(test["downgrade"], clf.predict_proba(test[cols])[:, 1] > cut))

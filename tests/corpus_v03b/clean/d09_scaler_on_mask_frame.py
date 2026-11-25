# expect: none
# LA05 class: the scaler is fitted on a frame selected by a date mask (X[mask]), applied to the later rows.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_parquet("spy_feats.parquet")
X = df[["f1", "f2", "f3"]]
y = df["up_next"]
is_fit = df.index < "2016-01-01"
sc = StandardScaler().fit(X[is_fit])
svc = SVC(probability=True).fit(sc.transform(X[is_fit]), y[is_fit])
p = svc.predict_proba(sc.transform(X[~is_fit]))[:, 1]

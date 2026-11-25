# expect: LA05
# Twin of clean/d09: scaler fitted on all rows; model on the masked rows.
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_parquet("spy_feats.parquet")
X = df[["f1", "f2", "f3"]]
y = df["up_next"]
is_fit = df.index < "2016-01-01"
sc = StandardScaler().fit(X)
svc = SVC(probability=True).fit(sc.transform(X[is_fit]), y[is_fit])
p = svc.predict_proba(sc.transform(X[~is_fit]))[:, 1]

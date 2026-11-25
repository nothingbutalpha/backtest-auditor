# expect: LA05
# Twin of clean/d04: fitted on all rows, evaluated on the masked later rows.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("panel.parquet")
feats = ["mom", "rev", "vol"]
mask = df["date"] < "2019-01-01"
clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=100)
clf.fit(df[feats], df["label"])
proba = clf.predict_proba(df.loc[~mask, feats])[:, 1]

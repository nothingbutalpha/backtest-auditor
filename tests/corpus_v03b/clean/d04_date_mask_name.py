# expect: none
# LA05 class: a boolean date mask defines the training rows; the complement is evaluated.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("panel.parquet")
feats = ["mom", "rev", "vol"]
mask = df["date"] < "2019-01-01"
clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=100)
clf.fit(df.loc[mask, feats], df.loc[mask, "label"])
proba = clf.predict_proba(df.loc[~mask, feats])[:, 1]

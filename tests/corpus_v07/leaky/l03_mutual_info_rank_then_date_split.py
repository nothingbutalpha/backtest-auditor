# expect: LA05|LA11@8
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("earnings_panel.parquet")
feats = [c for c in df.columns if c.startswith("f_")]
mi = pd.Series(mutual_info_classif(df[feats], df["beat_next"]), index=feats)
best = list(mi.sort_values(ascending=False).index[:12])
train, test = df[df["date"] < "2019-01-01"], df[df["date"] >= "2019-01-01"]
rf = RandomForestClassifier().fit(train[best], train["beat_next"])
print(rf.score(test[best], test["beat_next"]))

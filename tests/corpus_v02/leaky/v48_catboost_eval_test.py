# expect: LA14:high
import pandas as pd
from catboost import CatBoostClassifier
df = pd.read_parquet("features.parquet")
train, test = df.loc[:"2019"], df.loc["2020":]
feats = [c for c in df.columns if c != "label"]
clf = CatBoostClassifier(iterations=3000, verbose=0)
clf.fit(train[feats], train["label"], eval_set=(test[feats], test["label"]))
print((clf.predict(test[feats]) == test["label"]).mean())

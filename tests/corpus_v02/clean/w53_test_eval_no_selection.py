# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
train, test = df.loc[:"2018"], df.loc["2019":]
feats = ["mom_20", "vol_20", "rev_5"]
results = {}
for c in [0.01, 0.1, 1.0]:
    m = LogisticRegression(C=c).fit(train[feats], train["up"])
    results[c] = m.score(test[feats], test["up"])
print(pd.Series(results))

# expect: LA09:medium
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def standardize(s):
    return (s - s.mean()) / s.std()


px = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
ret = px.pct_change()
feats = pd.DataFrame({
    "mom": px.pct_change(60),
    "rev": -px.pct_change(5),
    "vol": ret.rolling(20).std(),
})
y = (ret.shift(-1) > 0).astype(int)
X = feats.apply(standardize)
data = X.join(y.rename("y")).dropna()
cut = data.index[int(len(data) * 0.8)]
clf = GradientBoostingClassifier().fit(data.loc[:cut, ["mom", "rev", "vol"]], data.loc[:cut, "y"])
print(clf.score(data.loc[cut:, ["mom", "rev", "vol"]], data.loc[cut:, "y"]))

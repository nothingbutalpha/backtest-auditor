# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression

def make_target(px, h=5):
    fwd = px.shift(-h) / px - 1
    return (fwd > 0).astype(int)

px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
X = pd.concat({"mom": px.pct_change(20), "vol": px.pct_change().rolling(20).std()}, axis=1).dropna()
y = make_target(px).reindex(X.index)
cut = "2018"
clf = LogisticRegression().fit(X.loc[:cut].iloc[:-5], y.loc[:cut].iloc[:-5])
print(clf.score(X.loc["2019":].iloc[:-5], y.loc["2019":].iloc[:-5]))

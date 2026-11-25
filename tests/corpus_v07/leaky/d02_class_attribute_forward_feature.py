# expect: LA01|LA10@9
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Features:
    def __init__(self, px):
        self.mom = px.pct_change(20)
        self.vol = px.pct_change().rolling(20).std()
        self.next_gap = px.shift(-1) / px - 1

    def matrix(self):
        return pd.concat({"mom": self.mom, "vol": self.vol, "gap": self.next_gap}, axis=1)

px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
X = Features(px).matrix().dropna()
y = (px.pct_change().shift(-1).reindex(X.index) > 0).astype(int)
clf = LogisticRegression().fit(X.loc[:"2018"], y.loc[:"2018"])
print(clf.score(X.loc["2019":], y.loc["2019":]))

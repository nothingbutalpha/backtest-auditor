# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Dataset:
    def __init__(self, px):
        self.mom = px.pct_change(20)
        self.vol = px.pct_change().rolling(20).std()
        self.target = (px.shift(-1) > px).astype(int)

    def frame(self):
        return pd.concat({"mom": self.mom, "vol": self.vol, "y": self.target}, axis=1).dropna()

px = pd.read_csv("spy.csv", index_col=0, parse_dates=True)["close"]
d = Dataset(px).frame()
cols = ["mom", "vol"]
clf = LogisticRegression().fit(d.loc[:"2018", cols], d.loc[:"2018", "y"])
print(clf.score(d.loc["2019":, cols], d.loc["2019":, "y"]))

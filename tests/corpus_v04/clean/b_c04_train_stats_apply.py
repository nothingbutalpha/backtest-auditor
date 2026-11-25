# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression

px = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
ret = px.pct_change()
feats = pd.DataFrame({
    "mom": px.pct_change(60),
    "rev": -px.pct_change(5),
    "vol": ret.rolling(20).std(),
})
y = (ret.shift(-1) > 0).astype(int)
train = feats.loc[:"2017-12-31"]
mu, sd = train.mean(), train.std()
X = feats.apply(lambda c: (c - mu[c.name]) / sd[c.name])
data = X.join(y.rename("y")).dropna()
model = LogisticRegression().fit(data.loc[:"2017-12-31", ["mom", "rev", "vol"]], data.loc[:"2017-12-31", "y"])
print(model.score(data.loc["2018-01-02":, ["mom", "rev", "vol"]], data.loc["2018-01-02":, "y"]))

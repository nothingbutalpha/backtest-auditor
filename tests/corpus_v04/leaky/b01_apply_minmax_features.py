# expect: LA09:medium
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
feats = feats.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
data = feats.join(y.rename("y")).dropna()
split = int(len(data) * 0.7)
model = LogisticRegression().fit(data.iloc[:split, :3], data["y"].iloc[:split])
print(model.score(data.iloc[split:, :3], data["y"].iloc[split:]))

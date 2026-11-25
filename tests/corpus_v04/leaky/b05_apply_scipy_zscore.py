# expect: LA09:medium
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import Ridge

px = pd.read_csv("spy.csv", index_col="date", parse_dates=True)["close"]
ret = px.pct_change()
feats = pd.DataFrame({
    "mom": px.pct_change(60),
    "rev": -px.pct_change(5),
    "vol": ret.rolling(20).std(),
})
y = (ret.shift(-1) > 0).astype(int)
X = feats.dropna().apply(zscore)
target = ret.shift(-1).reindex(X.index)
n = int(len(X) * 0.75)
model = Ridge().fit(X.iloc[:n], target.iloc[:n].fillna(0))
print(model.score(X.iloc[n:], target.iloc[n:].fillna(0)))

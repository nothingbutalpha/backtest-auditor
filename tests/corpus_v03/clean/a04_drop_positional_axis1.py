# expect: none
# Class 1: positional drop with axis=1.
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
data["fwd_ret"] = data["close"].pct_change().shift(-1)
data["target"] = data["fwd_ret"]
data["mom_10"] = data["close"].pct_change(10)
data["vol_10"] = data["close"].pct_change().rolling(10).std()
data = data.dropna()
X = data.drop(["fwd_ret", "target", "close"], axis=1)
y = data["target"]
split = int(len(data) * 0.7)
reg = Ridge(alpha=1.0).fit(X.iloc[:split], y.iloc[:split])
pred = reg.predict(X.iloc[split:])

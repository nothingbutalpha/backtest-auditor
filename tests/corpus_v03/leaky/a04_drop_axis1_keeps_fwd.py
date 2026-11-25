# expect: LA10:high
# Twin of clean/a04: drops only the target.
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_csv("spy.csv", parse_dates=["date"], index_col="date")
data["fwd_ret"] = data["close"].pct_change().shift(-1)
data["target"] = (data["fwd_ret"] > 0).astype(int)
data["mom_10"] = data["close"].pct_change(10)
data = data.dropna()
X = data.drop(["target"], axis=1)
y = data["target"]
split = int(len(data) * 0.7)
reg = Ridge(alpha=1.0).fit(X.iloc[:split], y.iloc[:split])

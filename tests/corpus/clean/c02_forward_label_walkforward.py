# expect: none
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["fwd_ret"] = df["close"].shift(-1) / df["close"] - 1
df["mom"] = df["close"].pct_change(20)
df["vol"] = df["close"].pct_change().rolling(20).std()
df = df.dropna()
features = ["mom", "vol"]
X = df[features]
y = df["fwd_ret"]
tscv = TimeSeriesSplit(n_splits=5, gap=1)
preds = []
for train_idx, test_idx in tscv.split(X):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds.append(pd.Series(model.predict(X.iloc[test_idx]), index=X.index[test_idx]))
oos = pd.concat(preds)

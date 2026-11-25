# expect: LA01|LA10:high
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
pv = (df["close"] * df["volume"]).rolling(20).sum()
vwap20 = pv / df["volume"].rolling(20).sum()
df["mom"] = df["close"].pct_change(20)
df["will_fill"] = (df["low"].shift(-1) <= df["close"] * 0.99).astype(int)
df["fwd_ret"] = df["close"].shift(-5) / df["close"] - 1
data = df.dropna()
X = data[["mom", "will_fill"]]
y = (data["fwd_ret"] > 0).astype(int)
model = LogisticRegression().fit(X.iloc[:2000], y.iloc[:2000])
print(model.score(X.iloc[2005:], y.iloc[2005:]))

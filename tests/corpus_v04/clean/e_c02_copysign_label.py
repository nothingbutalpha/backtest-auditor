# expect: none
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("spy.csv", index_col="date", parse_dates=True)
df["fwd_ret"] = df["close"].pct_change().shift(-1)
df["mom"] = df["close"].pct_change(20)
df["vol"] = df["close"].pct_change().rolling(20).std()
df = df.dropna()
y = np.copysign(1.0, df["fwd_ret"])
model = LogisticRegression().fit(df[["mom", "vol"]].iloc[:2000], y.iloc[:2000])
print(model.score(df[["mom", "vol"]].iloc[2001:], y.iloc[2001:]))

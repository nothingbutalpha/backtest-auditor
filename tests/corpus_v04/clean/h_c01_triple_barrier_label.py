# expect: none
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def triple_barrier(close, horizon=10, up=0.03, dn=0.03):
    """+1 if the upper barrier is hit first within `horizon` bars, -1 if the lower one is, else 0."""
    labels = np.zeros(len(close))
    for i in range(len(close) - horizon - 1):
        path = close.iloc[i + 1: i + 1 + horizon] / close.iloc[i] - 1
        hit_up = np.flatnonzero(path.to_numpy() >= up)
        hit_dn = np.flatnonzero(path.to_numpy() <= -dn)
        if len(hit_up) and (not len(hit_dn) or hit_up[0] < hit_dn[0]):
            labels[i] = 1
        elif len(hit_dn):
            labels[i] = -1
    return pd.Series(labels, index=close.index)


df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
df["mom"] = df["close"].pct_change(20)
df["vol"] = df["close"].pct_change().rolling(20).std()
y = triple_barrier(df["close"])
X = df[["mom", "vol"]].dropna()
y = y.reindex(X.index)
n = int(len(X) * 0.7)
model = RandomForestClassifier(n_estimators=200).fit(X.iloc[:n], y.iloc[:n])
print(model.score(X.iloc[n + 10:], y.iloc[n + 10:]))

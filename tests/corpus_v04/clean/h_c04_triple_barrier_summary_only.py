# expect: none
"""Label diagnostics before any model: class balance of triple-barrier labels."""
import pandas as pd

def triple_barrier_labels(close, vol, horizon=10, k=1.5):
    """+1 / -1 / 0: which volatility-scaled barrier the next `horizon` closes touch first."""
    labels = pd.Series(0, index=close.index, dtype=int)
    for i in range(len(close) - horizon):
        up = close.iloc[i] * (1 + k * vol.iloc[i])
        dn = close.iloc[i] * (1 - k * vol.iloc[i])
        path = close.iloc[i + 1: i + 1 + horizon]
        hit_up = path[path >= up]
        hit_dn = path[path <= dn]
        if len(hit_up) and (not len(hit_dn) or hit_up.index[0] < hit_dn.index[0]):
            labels.iloc[i] = 1
        elif len(hit_dn):
            labels.iloc[i] = -1
    return labels.iloc[: len(close) - horizon]


df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
df["vol"] = df["close"].pct_change().rolling(20).std()
df["mom"] = df["close"].pct_change(20)
X = df[["mom", "vol"]].dropna()
y = triple_barrier_labels(df["close"], df["vol"])
balance = y.loc[y.index.intersection(X.index)].value_counts(normalize=True)
print(balance)
print(y.describe(), y.mean())
y.to_csv("tb_labels.csv")

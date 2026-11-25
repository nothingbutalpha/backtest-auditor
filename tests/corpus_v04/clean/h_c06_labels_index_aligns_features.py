# expect: none
"""Feature table aligned to the labels' dates, then the class balance on the rows that survive."""
import pandas as pd

def triple_barrier_labels(close, vol, horizon=10, k=1.5):
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


px = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
close = px["close"]
vol = close.pct_change().rolling(20).std()
y = triple_barrier_labels(close, vol)
X = pd.DataFrame({"r5": close.pct_change(5), "r20": close.pct_change(20), "vol": vol}).loc[y.index].dropna()
common = X.index.intersection(y.index)
print(len(y), y.shape, len(common))
print(y.loc[X.index].value_counts(normalize=True))
X2 = X.reindex(y.index)
print(X2.describe())

# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

def standardize(frame):
    mu, sd = frame.mean(), frame.std()
    return (frame - mu) / sd, mu, sd

raw = pd.read_parquet("fx_features.parquet").sort_index()
cols = ["carry", "mom", "value"]
preds = []
for k in range(500, len(raw), 21):
    tr = raw.iloc[:k]
    te = raw.iloc[k:k + 21]
    Xtr, mu, sd = standardize(tr[cols])
    m = Ridge().fit(Xtr, tr["fwd_1m"])
    preds.append(pd.Series(m.predict((te[cols] - mu) / sd), index=te.index))
print(pd.concat(preds).describe())

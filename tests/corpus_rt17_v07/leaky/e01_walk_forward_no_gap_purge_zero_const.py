# leak: 11
import pandas as pd
from sklearn.linear_model import LogisticRegression
H = 10
PURGE = 0
f = pd.read_parquet("es_feats.parquet")
f["lbl"] = (f["close"].shift(-H) > f["close"]).astype(int)
f = f.dropna()
hits = []
for k in range(500, len(f) - 50, 50):
    a = f.iloc[: k - PURGE]
    b = f.iloc[k : k + 50]
    m = LogisticRegression().fit(a[["x1", "x2"]], a["lbl"])
    hits.append((m.predict(b[["x1", "x2"]]) == b["lbl"]).mean())
print(sum(hits) / len(hits))

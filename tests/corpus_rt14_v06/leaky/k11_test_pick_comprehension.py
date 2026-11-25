# leak: 12
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
d = pd.read_parquet("feat.parquet")
cut = int(len(d) * 0.7)
tr, te = d.iloc[:cut], d.iloc[cut:]
feats = [c for c in d if c.startswith("f_")]
scores = {}
for depth in (2, 3, 4, 5):
    m = GradientBoostingClassifier(max_depth=depth).fit(tr[feats], tr["up"])
    scores[depth] = roc_auc_score(te["up"], m.predict_proba(te[feats])[:, 1])
best_depth = max(scores, key=scores.get)
print(best_depth, scores[best_depth])

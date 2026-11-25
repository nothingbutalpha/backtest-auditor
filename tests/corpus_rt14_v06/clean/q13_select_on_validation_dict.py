# expect: none
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
d = pd.read_parquet("feat.parquet")
a, b = int(len(d) * 0.6), int(len(d) * 0.8)
tr, va, te = d.iloc[:a], d.iloc[a:b], d.iloc[b:]
feats = [c for c in d if c.startswith("f_")]
val_scores = {}
for depth in (2, 3, 4, 5):
    m = GradientBoostingClassifier(max_depth=depth).fit(tr[feats], tr["up"])
    val_scores[depth] = roc_auc_score(va["up"], m.predict_proba(va[feats])[:, 1])
best_depth = max(val_scores, key=val_scores.get)
final = GradientBoostingClassifier(max_depth=best_depth).fit(pd.concat([tr, va])[feats], pd.concat([tr, va])["up"])
print(roc_auc_score(te["up"], final.predict_proba(te[feats])[:, 1]))

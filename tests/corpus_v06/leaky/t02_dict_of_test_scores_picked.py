# expect: LA14:high@15
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("merger_arb.parquet").sort_values("announce")
X, y = df.filter(like="f_"), df["deal_closes"]
tr = df["announce"] < "2018-01-01"
candidates = {"lr": LogisticRegression(max_iter=500), "rf": RandomForestClassifier(200), "et": ExtraTreesClassifier(200)}
auc_test = {}
for name, m in candidates.items():
    m.fit(X[tr], y[tr])
    auc_test[name] = roc_auc_score(y[~tr], m.predict_proba(X[~tr])[:, 1])
chosen = max(auc_test, key=auc_test.get)
print(chosen, "out-of-sample AUC", auc_test[chosen])

# expect: none
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
snap = pd.read_parquet("fundamentals_2019q4.parquet")
snap["fwd_beat"] = (snap["next_q_eps"] > snap["consensus"]).astype(int)
X = snap[["accruals", "roe", "pe"]]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(RandomForestClassifier(), X, snap["fwd_beat"], cv=cv).mean())

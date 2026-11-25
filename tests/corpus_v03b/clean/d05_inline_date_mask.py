# expect: none
# LA05 class: frames filtered by an inline date comparison.
import pandas as pd
from sklearn.linear_model import LogisticRegression

CUT = "2018-12-31"
df = pd.read_parquet("panel.parquet")
feats = ["mom", "value"]
early = df[df["date"] <= CUT]
late = df[df["date"] > CUT]
lr = LogisticRegression(max_iter=1000)
lr.fit(early[feats], early["label"])
late = late.assign(p=lr.predict_proba(late[feats])[:, 1])

# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
d = pd.read_parquet("spy_feats.parquet")
d["next_up"] = (d["close"].shift(-1) > d["close"]).astype(int)
d = d.dropna()
print(cross_val_score(LogisticRegression(), d[["rsi", "mom"]], d["next_up"], cv=5).mean())

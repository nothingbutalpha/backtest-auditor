# expect: none
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("ust10y_features.parquet").sort_index()
X = df[["slope", "carry", "mom_60", "vol_20"]]
y = (df["yld_chg_fwd_20"] > 0).astype(int)
auc = cross_val_score(RandomForestClassifier(), X, y, cv=TimeSeriesSplit(10, gap=20), scoring="roc_auc")
print(auc.mean())

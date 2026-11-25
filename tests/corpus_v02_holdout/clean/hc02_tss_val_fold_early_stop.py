# expect: none
# Looks like test-set early stopping, but stops on a TimeSeriesSplit validation fold; test is scored once.
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

df = pd.read_parquet("features_daily.parquet")
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.iloc[:-1]
features = ["mom_5", "mom_20", "vol_20", "rsi_14"]

test_start = int(len(df) * 0.8)
dev, test = df.iloc[:test_start], df.iloc[test_start:]
train_idx, val_idx = list(TimeSeriesSplit(n_splits=5).split(dev))[-1]
tr, va = dev.iloc[train_idx], dev.iloc[val_idx]

model = XGBClassifier(n_estimators=2000, learning_rate=0.02, max_depth=3, early_stopping_rounds=50)
model.fit(tr[features], tr["target"], eval_set=[(va[features], va["target"])], verbose=False)

auc = roc_auc_score(test["target"], model.predict_proba(test[features])[:, 1])
print("Held-out test AUC:", round(auc, 3))

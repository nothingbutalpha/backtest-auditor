# expect: none
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
HORIZON = 21
f = pd.read_parquet("es_feats.parquet")
f["y"] = f["close"].pct_change(HORIZON).shift(-HORIZON)
f = f.dropna()
cv = TimeSeriesSplit(n_splits=5, gap=HORIZON)
print(cross_val_score(Ridge(), f[["x1", "x2"]], f["y"], cv=cv).mean())

# expect: LA06:medium@7
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

FEATS = ["basis", "oi_chg", "funding"]
data = pd.read_parquet("perp_features.parquet").dropna().sample(frac=1.0, random_state=3)
for tr, te in TimeSeriesSplit(n_splits=5).split(data):
    m = Ridge().fit(data[FEATS].iloc[tr], data["fwd_ret_4h"].iloc[tr])
    print(m.score(data[FEATS].iloc[te], data["fwd_ret_4h"].iloc[te]))

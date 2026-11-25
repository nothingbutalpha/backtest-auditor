# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

feats = pd.read_parquet("macro_features_monthly.parquet")
feats = feats.dropna(axis=1, thresh=int(0.9 * len(feats)))
y = pd.read_parquet("spx_monthly.parquet")["fwd_ret_1m"].reindex(feats.index)
data = feats.join(y).dropna()
m = Ridge().fit(data.drop(columns="fwd_ret_1m").loc[:"2015"], data["fwd_ret_1m"].loc[:"2015"])
print(m.score(data.drop(columns="fwd_ret_1m").loc["2016-02":], data["fwd_ret_1m"].loc["2016-02":]))

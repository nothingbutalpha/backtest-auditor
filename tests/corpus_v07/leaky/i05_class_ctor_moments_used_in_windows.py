# expect: LA09|LA05@7
import pandas as pd
from sklearn.linear_model import ElasticNet

class Standardizer:
    def __init__(self, data: pd.DataFrame):
        self.center = data.mean()
        self.scale = data.std()

    def apply(self, block: pd.DataFrame) -> pd.DataFrame:
        return (block - self.center) / self.scale

feats = pd.read_parquet("commodity_factors.parquet")
y = pd.read_parquet("commodity_targets.parquet")["fwd_1m"]
std = Standardizer(feats)
tr, te = std.apply(feats.loc[:"2017"]), std.apply(feats.loc["2018":])
model = ElasticNet(alpha=1e-3).fit(tr, y.loc[tr.index])
print(model.score(te, y.loc[te.index]))

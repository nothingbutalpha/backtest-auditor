# expect: none
import pandas as pd
from sklearn.linear_model import ElasticNet

class Standardizer:
    def __init__(self, data):
        self.center = data.mean()
        self.scale = data.std()

    def apply(self, block):
        return (block - self.center) / self.scale

feats = pd.read_parquet("commodity_factors.parquet")
y = pd.read_parquet("commodity_targets.parquet")["fwd_1m"]
train, test = feats.loc[:"2017"], feats.loc["2018":]
std = Standardizer(train)
model = ElasticNet(alpha=1e-3).fit(std.apply(train), y.loc[train.index])
print(model.score(std.apply(test), y.loc[test.index]))

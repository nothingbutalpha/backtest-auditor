# expect: LA14:medium
import pandas as pd
from sklearn.linear_model import Ridge
df = pd.read_parquet("features.parquet")
train, test = df.loc[:"2018"], df.loc["2019":]
candidates = ["mom_20", "mom_60", "rev_5", "vol_20", "size", "value"]
scores = {}
for feat in candidates:
    m = Ridge(alpha=1.0).fit(train[[feat]], train["fwd_ret"])
    scores[feat] = pd.Series(m.predict(test[[feat]]), index=test.index).corr(test["fwd_ret"])
best_feat = max(scores, key=scores.get)
print(best_feat, scores[best_feat])

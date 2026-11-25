# expect: LA09|LA05@6
import pandas as pd
from sklearn.linear_model import Ridge

def clip_tails(frame, q=0.02):
    bounds = frame.quantile([q, 1 - q])
    return frame.clip(bounds.iloc[0], bounds.iloc[1], axis=1)

raw = pd.read_parquet("rates_features.parquet").sort_index()
feats = ["slope", "curv", "carry_3m", "mom_6m"]
X = clip_tails(raw[feats])
y = raw["fwd_ret_21"]
out = []
for k in range(750, len(X), 63):
    m = Ridge(alpha=3.0).fit(X.iloc[:k], y.iloc[:k])
    out.append(pd.Series(m.predict(X.iloc[k:k + 63]), index=X.index[k:k + 63]))
print(pd.concat(out).corr(y))

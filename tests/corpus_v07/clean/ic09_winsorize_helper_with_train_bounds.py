# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

def clip_to(frame, lo, hi):
    return frame.clip(lo, hi, axis=1)

raw = pd.read_parquet("rates_features.parquet").sort_index()
feats = ["slope", "curv", "carry_3m"]
train, test = raw.loc[:"2016"], raw.loc["2017":]
lo, hi = train[feats].quantile(0.02), train[feats].quantile(0.98)
m = Ridge().fit(clip_to(train[feats], lo, hi), train["fwd_ret_21"])
print(m.score(clip_to(test[feats], lo, hi), test["fwd_ret_21"]))

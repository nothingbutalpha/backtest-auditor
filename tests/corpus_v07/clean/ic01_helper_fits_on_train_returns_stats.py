# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

def fit_moments(frame):
    return frame.mean(), frame.std()

raw = pd.read_parquet("rates_features.parquet").sort_index()
feats = ["slope", "curv", "carry_3m"]
train, test = raw.loc[:"2016"], raw.loc["2017":]
mu, sd = fit_moments(train[feats])
m = Ridge().fit((train[feats] - mu) / sd, train["fwd_ret_21"])
print(m.score((test[feats] - mu) / sd, test["fwd_ret_21"]))

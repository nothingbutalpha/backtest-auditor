# expect: none
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
def fit_scaler(frame):
    return StandardScaler().fit(frame)
raw = pd.read_parquet("btc_features.parquet")
target = (raw["close"].shift(-1) > raw["close"]).astype(int)
cols = ["rsi", "atr", "obv"]
early = raw.index < "2021-01-01"
sc = fit_scaler(raw.loc[early, cols])
clf = LogisticRegression().fit(sc.transform(raw.loc[early, cols]), target[early])
print((clf.predict(sc.transform(raw.loc[~early, cols])) == target[~early]).mean())

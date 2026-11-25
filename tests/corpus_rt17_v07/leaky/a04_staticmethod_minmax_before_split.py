# leak: 7
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
class Prep:
    @staticmethod
    def squash(frame):
        return (frame - frame.min()) / (frame.max() - frame.min())
raw = pd.read_parquet("btc_features.parquet")
target = (raw["close"].shift(-1) > raw["close"]).astype(int)
X = Prep.squash(raw[["rsi", "atr", "obv"]])
n = int(len(X) * 0.8)
clf = GradientBoostingClassifier().fit(X.iloc[:n], target.iloc[:n])
print((clf.predict(X.iloc[n:]) == target.iloc[n:]).mean())

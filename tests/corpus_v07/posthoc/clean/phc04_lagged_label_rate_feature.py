# expect: none
import pandas as pd
from sklearn.linear_model import LogisticRegression

H = 10
df = pd.read_csv("gold_daily.csv", index_col=0, parse_dates=True)
df["label"] = (df["close"].shift(-H) > df["close"]).astype(int)
df["label_rate"] = df["label"].shift(H + 1).rolling(20).mean()

df["mom"] = df["close"].pct_change(20)
df = df.dropna()
train, test = df.loc[:"2017"].iloc[:-H], df.loc["2018":]
cols = ["label_rate", "mom"]
lr = LogisticRegression().fit(train[cols], train["label"])
print(lr.score(test[cols], test["label"]))

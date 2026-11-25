import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["target"] = (df["ret"].shift(-1) > 0).astype(int)
df = df.dropna()
train = df.loc[:"2018-12-31"]
test = df.loc["2019-01-01":]
scaler = StandardScaler().fit(train[["ret"]])
clf = LogisticRegression().fit(scaler.transform(train[["ret"]]), train["target"])
print(clf.score(scaler.transform(test[["ret"]]), test["target"]))

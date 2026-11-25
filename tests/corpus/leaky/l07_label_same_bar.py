# expect: LA08:possible
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["ret"] = df["close"].pct_change()
df["mom_20"] = df["close"].pct_change(20)
df["vol_20"] = df["ret"].rolling(20).std()
df["target"] = df["ret"]
df = df.dropna()
train, test = df.loc[:"2019"], df.loc["2020":]
model = Ridge().fit(train[["mom_20", "vol_20"]], train["target"])
print(model.score(test[["mom_20", "vol_20"]], test["target"]))

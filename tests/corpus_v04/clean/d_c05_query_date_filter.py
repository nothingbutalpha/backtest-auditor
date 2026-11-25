# expect: none
import pandas as pd
from sklearn.linear_model import Ridge

df = pd.read_parquet("features.parquet")
df["fwd_ret"] = df.groupby("ticker")["close"].shift(-5) / df["close"] - 1
train = df.query("date < '2019-01-01'").dropna()
test = df.query("date >= '2019-01-08'").dropna()
cols = ["mom", "value", "quality"]
model = Ridge().fit(train[cols], train["fwd_ret"])
test["pred"] = model.predict(test[cols])
print(test[["pred", "fwd_ret"]].corr().iloc[0, 1])

# expect: LA10@10
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("gold_daily.parquet")
df["r1"] = df["close"].pct_change()
df["ret_fwd_10"] = df["close"].shift(-10) / df["close"] - 1
df["y"] = (df["ret_fwd_10"] > 0).astype(int)
df = df.dropna()
predictors = ["r1", "atr_14", "ret_fwd_10"]
cut = int(len(df) * 0.8)
model = RandomForestClassifier(n_estimators=200)
train = df.iloc[:cut]
model.fit(train[predictors], train["y"])
print(model.score(df[predictors].iloc[cut:], df["y"].iloc[cut:]))

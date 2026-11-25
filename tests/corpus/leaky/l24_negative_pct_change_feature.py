# expect: LA01:high, LA10:high
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date")
df["mom_5"] = df["close"].pct_change(5)
df["feat_next_ret"] = df["close"].pct_change(-1)
df["label"] = (df["close"].shift(-5) > df["close"]).astype(int)
df = df.dropna()
train = df.loc[:"2019"]
model = LogisticRegression().fit(train[["mom_5", "feat_next_ret"]], train["label"])

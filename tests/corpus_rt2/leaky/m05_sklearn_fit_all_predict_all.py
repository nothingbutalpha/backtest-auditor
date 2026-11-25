import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["r1"] = df["Close"].pct_change()
df["r5"] = df["Close"].pct_change(5)
df["up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()
clf = LogisticRegression().fit(df[["r1", "r5"]], df["up"])
df["signal"] = clf.predict(df[["r1", "r5"]])
df["strategy"] = df["signal"] * df["r1"].shift(-1)

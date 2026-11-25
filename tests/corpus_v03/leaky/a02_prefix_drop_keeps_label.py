# expect: LA10:high
# Twin of clean/a02: the comprehension drops fwd_* but the label is kept in X.
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("daily.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
px = df.groupby("ticker")["close"]
df["fwd_close"] = px.shift(-5)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > 0).astype(int)
df["mom"] = px.pct_change(60)
df = df.dropna()
fwd_cols = [c for c in df.columns if c.startswith("fwd_")]
X = df.drop(columns=fwd_cols + ["date", "ticker"])
y = df["label"]
cut = df["date"] < "2020-01-01"
clf = LogisticRegression(max_iter=500).fit(X[cut], y[cut])

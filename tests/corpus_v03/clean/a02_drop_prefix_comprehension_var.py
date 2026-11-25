# expect: none
# Class 1: the drop list is built by a comprehension over df.columns filtered by prefix.
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("daily.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
px = df.groupby("ticker")["close"]
df["fwd_close"] = px.shift(-5)
df["fwd_ret"] = df["fwd_close"] / df["close"] - 1
df["label"] = (df["fwd_ret"] > 0).astype(int)
df["rsi_like"] = px.pct_change().rolling(14).mean()
df["mom"] = px.pct_change(60)
df = df.dropna()
fwd_cols = [c for c in df.columns if c.startswith("fwd_")]
X = df.drop(columns=fwd_cols + ["label", "date", "ticker"])
y = df["label"]
cut = df["date"] < "2020-01-01"
clf = LogisticRegression(max_iter=500).fit(X[cut], y[cut])

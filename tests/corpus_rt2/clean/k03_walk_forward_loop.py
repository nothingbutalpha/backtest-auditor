import pandas as pd
from sklearn.linear_model import Ridge
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["mom"] = df["Close"].pct_change(20)
df["fwd_ret"] = df["ret"].shift(-1)
df = df.dropna()
preds = pd.Series(index=df.index, dtype=float)
for i in range(500, len(df)):
    train = df.iloc[i - 500:i - 1]
    m = Ridge().fit(train[["mom"]], train["fwd_ret"])
    preds.iloc[i] = m.predict(df[["mom"]].iloc[[i]])[0]
position = (preds > 0).astype(int)
strategy = position * df["fwd_ret"] - 0.0005 * position.diff().abs()

import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["vol"] = df["Close"].pct_change().rolling(20).std()
train = df.loc[:"2015"]
threshold = train["vol"].quantile(0.8)
test = df.loc["2016":].copy()
test["position"] = (test["vol"] < threshold).astype(int)
test["strat"] = test["position"].shift(1) * test["Close"].pct_change() - 0.0005 * test["position"].diff().abs()

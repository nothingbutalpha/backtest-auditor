import pandas as pd
df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
df["fwd_ret"] = df["close"].pct_change().shift(-1)
corrs = df.drop(columns=["fwd_ret"]).corrwith(df["fwd_ret"]).abs()
best = corrs.nlargest(5).index
train = df.loc[:"2018"]
test = df.loc["2019":]

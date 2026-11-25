import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df = df.dropna()
scaler = MinMaxScaler()
feats = scaler.fit_transform(df[["ret"]])
cut = int(0.8 * len(df))
m = LinearRegression().fit(feats[:cut], df["fwd_ret"][:cut])
print(m.score(feats[cut:], df["fwd_ret"][cut:]))

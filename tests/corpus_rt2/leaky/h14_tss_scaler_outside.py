import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["ret"] = df["Close"].pct_change()
df["fwd_ret"] = df["ret"].shift(-1)
df = df.dropna()
X = StandardScaler().fit_transform(df[["ret"]])
y = df["fwd_ret"].values
for tr, te in TimeSeriesSplit(5).split(X):
    m = Ridge().fit(X[tr], y[tr])
    print(m.score(X[te], y[te]))

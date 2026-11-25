import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
ret = df["Close"].pct_change()
mu = ret.expanding(252).mean()
sd = ret.expanding(252).std()
df["z"] = (ret - mu) / sd
df["position"] = (df["z"] < -2).astype(int)
df["pnl"] = df["position"].shift(1) * ret - 0.0005 * df["position"].diff().abs()

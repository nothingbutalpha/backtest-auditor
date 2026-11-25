# leak: 4
import pandas as pd
rets = pd.read_csv("sp500_returns.csv", index_col=0, parse_dates=True)
rets = rets.dropna(axis="columns", how="any")
w = rets.rolling(60).std().rdiv(1)
w = w.div(w.sum(axis=1), axis=0).shift(1)
print((w * rets).sum(axis=1).mean())

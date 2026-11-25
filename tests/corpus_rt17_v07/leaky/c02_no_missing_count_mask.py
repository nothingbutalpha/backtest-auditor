# leak: 4
import pandas as pd
closes = pd.read_csv("smallcaps.csv", index_col=0, parse_dates=True)
closes = closes.loc[:, closes.isna().sum() == 0]
r = closes.pct_change()
pos = (r.rolling(5).sum().rank(axis=1) <= 20).astype(float).shift(1)
print((pos * r).mean(axis=1).cumsum().iloc[-1])

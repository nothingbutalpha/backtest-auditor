import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
ret = df["Close"].pct_change()
vol = ret.ewm(span=60).std() * 252 ** 0.5
weight = (0.10 / vol).clip(upper=2).shift(1)
fees = weight.diff().abs() * 0.0002
strategy = weight * ret - fees

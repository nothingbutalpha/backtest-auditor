import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
rets = df["Close"].pct_change()
signal = (df["Close"] > df["Close"].ewm(span=30).mean()).astype(float)
strategy = signal.mul(rets)
print(strategy.sum())

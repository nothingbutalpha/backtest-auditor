# expect: none
import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
r = df["close"].pct_change()
strategy = (df["close"] > df["close"].rolling(50).mean()).astype(int).shift(1) * r - 0.0005
print(strategy.mean() * 252)

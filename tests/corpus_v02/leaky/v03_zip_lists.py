# expect: LA07:high
import pandas as pd
closes = pd.read_csv("spy.csv")["close"].tolist()
fee = 0.0005
signals = [1 if today > yday else 0 for yday, today in zip(closes[:-1], closes[1:])]
returns = [today / yday - 1 for yday, today in zip(closes[:-1], closes[1:])]
pnl = [s * r - fee * s for s, r in zip(signals, returns)]
print(sum(pnl))

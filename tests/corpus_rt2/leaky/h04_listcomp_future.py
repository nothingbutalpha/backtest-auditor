import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
closes = df["Close"].tolist()
signals = [1 if closes[i + 1] > closes[i] else 0 for i in range(len(closes) - 1)] + [0]
df["position"] = signals
df["strategy"] = df["position"].shift(1) * df["Close"].pct_change()

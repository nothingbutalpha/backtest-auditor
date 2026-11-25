import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
fee = 0.0005
trades = []
for i in range(50, len(df) - 1):
    ma = df["Close"].iloc[i - 50:i + 1].mean()
    if df["Close"].iloc[i] > ma:
        entry_price = df["Open"].iloc[i + 1]
        exit_price = df["Close"].iloc[i + 1]
        trades.append(exit_price / entry_price - 1 - 2 * fee)

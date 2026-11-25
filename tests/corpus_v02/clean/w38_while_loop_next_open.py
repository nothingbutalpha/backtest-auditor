# expect: none
import pandas as pd
df = pd.read_csv("spy.csv")
opens, closes = df["open"].to_numpy(), df["close"].to_numpy()
hi20 = df["high"].rolling(20).max().shift(1).to_numpy()
i, trades, fee = 20, [], 0.0005
while i < len(df) - 6:
    if closes[i] > hi20[i]:
        buy = opens[i + 1]
        trades.append(closes[i + 6] / buy - 1 - 2 * fee)
        i += 5
    i += 1
print(sum(trades))

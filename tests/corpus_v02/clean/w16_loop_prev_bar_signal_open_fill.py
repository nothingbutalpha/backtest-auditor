# expect: none
import pandas as pd
df = pd.read_csv("aapl.csv")
df["ema"] = df["close"].ewm(span=20).mean()
position, entry, pnl, commission = 0, 0.0, 0.0, 0.001
for i in range(1, len(df)):
    if position == 0 and df["close"].iloc[i - 1] > df["ema"].iloc[i - 1]:
        position, entry = 1, df["open"].iloc[i]
    elif position == 1 and df["close"].iloc[i - 1] < df["ema"].iloc[i - 1]:
        pnl += df["open"].iloc[i] / entry - 1 - 2 * commission
        position = 0
print(pnl)

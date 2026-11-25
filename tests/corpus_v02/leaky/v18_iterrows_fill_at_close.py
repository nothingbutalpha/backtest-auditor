# expect: LA07:medium
import pandas as pd
df = pd.read_csv("aapl.csv")
df["ema"] = df["close"].ewm(span=20).mean()
position, entry, pnl, commission = 0, 0.0, 0.0, 0.001
for idx, row in df.iterrows():
    if position == 0 and row["close"] > row["ema"]:
        position, entry = 1, row["close"]
    elif position == 1 and row["close"] < row["ema"]:
        pnl += row["close"] / entry - 1 - 2 * commission
        position = 0
print(pnl)

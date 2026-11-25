# expect: LA07:high
import pandas as pd

df = pd.read_csv("gld.csv", index_col=0, parse_dates=True)
signal = df["close"] < df["close"].rolling(10).min().shift(1)
in_pos, trades = False, []
for i in range(len(df) - 5):
    if signal.iloc[i] and not in_pos:
        entry_px = df["open"].iloc[i]
        exit_px = df["close"].iloc[i + 5]
        trades.append(exit_px / entry_px - 1 - 0.001)
print(sum(trades) / len(trades))

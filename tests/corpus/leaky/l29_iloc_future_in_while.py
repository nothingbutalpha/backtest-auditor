# expect: LA13:high
# From the red-team review of v0.1 (regression case).
import pandas as pd
df = pd.read_csv("spy.csv"); i = 0; trades = []
while i < len(df) - 1:
    if df["close"].iloc[i + 1] > df["close"].iloc[i] * 1.01:
        trades.append(df["close"].iloc[i + 1] / df["close"].iloc[i] - 1)
    i += 1

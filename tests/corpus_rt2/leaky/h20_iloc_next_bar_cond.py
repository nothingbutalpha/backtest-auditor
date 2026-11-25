import pandas as pd
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
sig = []
for i in range(len(df) - 1):
    if df["Close"].iloc[i + 1] > df["Open"].iloc[i + 1]:
        sig.append(1)
    else:
        sig.append(0)
sig.append(0)
df["sig"] = sig

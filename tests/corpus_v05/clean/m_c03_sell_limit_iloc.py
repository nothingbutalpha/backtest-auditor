# expect: none
import pandas as pd

df = pd.read_csv("qqq.csv")
df["rsi_up"] = df["close"].diff().clip(lower=0).rolling(14).mean()
rets = []
i = 14
while i < len(df) - 4:
    if df["rsi_up"].iloc[i] > 1.5 and df["close"].iloc[i] > df["close"].iloc[i - 1]:
        limit_px = df["close"].iloc[i] * 1.01
        if df["high"].iloc[i + 1] >= limit_px:
            fill_px = max(limit_px, df["open"].iloc[i + 1])
            cover_px = df["close"].iloc[i + 3]
            rets.append(fill_px / cover_px - 1 - 0.001)
    i += 1
print(sum(rets) / max(len(rets), 1))

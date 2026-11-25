# expect: LA07:high
import pandas as pd
df = pd.read_csv("spy.csv")
df["sma"] = df["close"].rolling(50).mean()
cash, shares, fee = 10000.0, 0, 1.0
for i in range(50, len(df)):
    if shares == 0 and df["close"].iloc[i] > df["sma"].iloc[i]:
        entry_price = df["open"].iloc[i]
        shares = int(cash // entry_price)
        cash -= shares * entry_price + fee
    elif shares > 0 and df["close"].iloc[i] < df["sma"].iloc[i]:
        cash += shares * df["open"].iloc[i + 1] - fee
        shares = 0
print(cash)

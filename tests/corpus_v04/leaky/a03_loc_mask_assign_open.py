# expect: LA07:high
import pandas as pd

df = pd.read_csv("aapl.csv", index_col=0, parse_dates=True)
df["sma50"] = df["close"].rolling(50).mean()
cross = (df["close"] > df["sma50"]) & (df["close"].shift(1) <= df["sma50"].shift(1))
df["entry_price"] = float("nan")
df.loc[cross, "entry_price"] = df["open"]
df["exit_price"] = df["close"].shift(-20)
trades = df.dropna(subset=["entry_price"])
trades["ret"] = trades["exit_price"] / trades["entry_price"] - 1 - 0.001
print(trades["ret"].describe())

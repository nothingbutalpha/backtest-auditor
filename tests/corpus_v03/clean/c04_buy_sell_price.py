# expect: none
# Class 3: buy and sell prices at the next opens.
import pandas as pd

df = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
df["position"] = (df["Close"] > df["Close"].rolling(100).mean()).astype(int)
buy_price = df["Open"].shift(-1)
sell_price = df["Open"].shift(-2)
df["trade_return"] = sell_price / buy_price - 1
df["strat"] = df["position"] * df["trade_return"] - 0.0005 * df["position"].diff().abs()
print(df["strat"].mean() * 252 / (df["strat"].std() * 252 ** 0.5))

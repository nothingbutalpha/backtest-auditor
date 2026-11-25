# expect: LA01|LA10:high
# Twin of clean/c04: momentum measured to tomorrow's open feeds the position.
import pandas as pd

df = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
buy_price = df["Open"].shift(-1)
sell_price = df["Open"].shift(-2)
df["mom"] = buy_price / df["Close"].shift(20) - 1
df["position"] = (df["mom"] > 0).astype(int)
df["strat"] = df["position"] * (sell_price / buy_price - 1) - 0.0005

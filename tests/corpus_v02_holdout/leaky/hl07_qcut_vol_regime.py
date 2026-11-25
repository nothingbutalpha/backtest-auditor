# expect: LA09
# Volatility regimes come from pd.qcut over the whole history, so bucket edges use future data.
import pandas as pd

df = pd.read_csv("spx.csv", parse_dates=["date"]).set_index("date")
df["ret"] = df["close"].pct_change()
df["vol20"] = df["ret"].rolling(20).std()
df["regime"] = pd.qcut(df["vol20"], 5, labels=False)

df["pos"] = (df["regime"] <= 1).astype(int).shift(1).fillna(0)
df["net"] = df["pos"] * df["ret"] - df["pos"].diff().abs().fillna(0) * 0.0005
print(df.groupby("regime")["net"].mean())
print("CAGR", (1 + df["net"]).prod() ** (252 / len(df)) - 1)

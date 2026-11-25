# expect: LA07:high
import pandas as pd
df = pd.read_csv("es.csv", index_col=0, parse_dates=True)
df["breakout"] = (df["close"] > df["high"].rolling(20).max().shift(1)).astype(int)
df["pnl"] = df["breakout"] * (df["close"] - df["open"]) * 50 - 2.5 * df["breakout"]
print(df["pnl"].sum())

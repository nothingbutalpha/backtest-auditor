# expect: none
# Looks like a full-sample qcut, but quintiles are formed per date inside groupby('date').
import pandas as pd

df = pd.read_csv("universe_daily.csv", parse_dates=["date"]).sort_values(["ticker", "date"])
df["ret"] = df.groupby("ticker")["close"].pct_change()
df["rev5"] = -df.groupby("ticker")["close"].pct_change(5)
df["q"] = df.groupby("date")["rev5"].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))

df["long"] = (df["q"] == 4).astype(float)
df["pos"] = df.groupby("ticker")["long"].shift(1).fillna(0)
df["net"] = df["pos"] * df["ret"] - 0.0015 * df.groupby("ticker")["pos"].diff().abs().fillna(0)
daily = df.groupby("date")["net"].sum() / df.groupby("date")["pos"].sum().clip(lower=1)
print("Mean daily net:", daily.mean())

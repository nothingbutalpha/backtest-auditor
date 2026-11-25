# expect: none
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import pandas as pd
df = pd.read_csv("es.csv", parse_dates=["date"]).set_index("date")
df["sma"] = df["close"].rolling(20).mean()
cash, pos, equity = 100_000.0, 0, []
fee = 2.5
for i in range(21, len(df)):
    prev = df.iloc[i - 1]
    bar = df.iloc[i]
    if pos == 0 and prev["close"] > prev["sma"]:
        pos = int(cash // bar["open"])
        cash -= pos * bar["open"] + fee
    elif pos > 0 and prev["close"] < prev["sma"]:
        cash += pos * bar["open"] - fee
        pos = 0
    equity.append(cash + pos * bar["close"])
eq = pd.Series(equity)
print("sharpe", eq.pct_change().mean() / eq.pct_change().std() * 252 ** 0.5)

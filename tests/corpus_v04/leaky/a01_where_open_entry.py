# expect: LA07:high
"""Mean-reversion: buy when RSI(14) closes below 30, priced at that day's open."""
import pandas as pd

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
df["rsi"] = rsi(df["close"])
entries = df["rsi"] < 30
entry_px = df["open"].where(entries)
exit_px = df["close"].shift(-5).where(entries)
trade_ret = (exit_px / entry_px - 1 - 0.001).dropna()
print(len(trade_ret), trade_ret.mean(), (trade_ret > 0).mean())

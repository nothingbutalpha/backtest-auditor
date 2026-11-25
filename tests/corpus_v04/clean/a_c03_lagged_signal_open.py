# expect: none
import pandas as pd

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

df = pd.read_csv("spy_daily.csv", index_col="date", parse_dates=True)
df["rsi"] = rsi(df["close"])
entries = (df["rsi"] < 30).shift(1, fill_value=False)
entry_px = df["open"].where(entries)
exit_px = df["open"].shift(-5).where(entries)
print(((exit_px / entry_px) - 1 - 0.001).mean())

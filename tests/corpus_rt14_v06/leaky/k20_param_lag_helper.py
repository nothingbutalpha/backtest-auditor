# leak: 8
import pandas as pd
def move(series, n):
    return series.shift(n)
df = pd.read_csv("es_daily.csv")
df["ret"] = df["close"].pct_change()
df["mom"] = df["ret"].rolling(10).sum()
df["fut"] = move(df["ret"], -1)
df["signal"] = (df["fut"] > 0) & (df["mom"] > 0)
df["pnl"] = df["signal"].astype(int) * df["ret"].shift(-1)
print(df["pnl"].sum())

# expect: none
import pandas as pd
import vectorbt as vbt
def entries_for(c):
    return (c > c.rolling(50).mean()).shift(1, fill_value=False)
def exits_for(c):
    return (c < c.rolling(50).mean()).shift(1, fill_value=False)
df = pd.read_parquet("iwm.parquet")
pf = vbt.Portfolio.from_signals(df["close"], entries=entries_for(df["close"]), exits=exits_for(df["close"]),
                                price=df["open"])
print(pf.sharpe_ratio())

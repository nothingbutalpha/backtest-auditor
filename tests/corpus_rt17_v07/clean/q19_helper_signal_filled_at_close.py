# expect: none
import pandas as pd
import vectorbt as vbt
def trend(c):
    return c.rolling(20).mean() > c.rolling(100).mean()
df = pd.read_parquet("xlk.parquet")
on = trend(df["close"])
pf = vbt.Portfolio.from_signals(df["close"], entries=on.shift(1, fill_value=False), exits=(~on).shift(1, fill_value=False))
print(pf.total_return())

# expect: none
import pandas as pd

df = pd.read_parquet("panel.parquet")
close = df.pivot_table(index="date", columns="ticker", values="close")
rets = close.pct_change()
signal = (close > close.rolling(200).mean()).astype(float).shift(1)
port = (signal * rets).sum(axis=1).div(signal.sum(axis=1)) - 0.0003
print(port.mean() * 252)

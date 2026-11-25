# expect: SB01:medium@5
import pandas as pd

closes = pd.read_parquet("stoxx600_closes.parquet").loc["2006":"2024"]
closes = closes.dropna(axis=1)
rets = closes.pct_change()
w = (rets.rolling(252).sum().shift(21).rank(axis=1, pct=True) > 0.8).astype(float)
w = w.div(w.sum(axis=1), axis=0).shift(1)
print((w * rets).sum(axis=1).mean() * 252)

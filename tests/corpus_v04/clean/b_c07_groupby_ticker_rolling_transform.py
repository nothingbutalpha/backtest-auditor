# expect: none
import pandas as pd

panel = pd.read_parquet("panel.parquet").sort_values(["ticker", "date"])
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
panel["mom"] = panel.groupby("ticker")["close"].pct_change(60)
panel["zmom"] = panel.groupby("ticker")["mom"].transform(lambda s: (s - s.rolling(252).mean()) / s.rolling(252).std())
panel["signal"] = panel.groupby("ticker")["zmom"].shift(1) > 1
print(panel.groupby("date").apply(lambda g: g.loc[g["signal"], "ret"].mean()).mean())

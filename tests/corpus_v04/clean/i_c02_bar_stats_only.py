# expect: none
import numpy as np
import pandas as pd

ticks = pd.read_parquet("es_ticks.parquet")  # columns: price, size; DatetimeIndex
ticks["pv"] = ticks["price"] * ticks["size"]
bars = ticks.resample("5min").agg({"price": ["first", "max", "min", "last"], "size": "sum"})
bars.columns = ["open", "high", "low", "close", "volume"]
bars = bars.dropna()
r = np.log(bars["close"]).diff()
print("5-min vol", r.std() * np.sqrt(78 * 252), "avg volume", bars["volume"].mean())

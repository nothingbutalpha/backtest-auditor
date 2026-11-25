# expect: none
import pandas as pd

ticks = pd.read_parquet("es_ticks.parquet")  # columns: price, size; DatetimeIndex
ticks["pv"] = ticks["price"] * ticks["size"]
ohlc = ticks["price"].resample("1min").ohlc()
vol = ticks["size"].resample("1min").sum().rename("volume")
vwap = (ticks["pv"].resample("1min").sum() / vol).rename("vwap")
bars = pd.concat([ohlc, vol, vwap], axis=1).dropna()
bars.to_csv("es_1min.csv")

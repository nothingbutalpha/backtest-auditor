# expect: none
import pandas as pd

ticks = pd.read_parquet("es_ticks.parquet")  # columns: price, size; DatetimeIndex
ticks["pv"] = ticks["price"] * ticks["size"]
bars = ticks["price"].resample("1min").ohlc()
bars["volume"] = ticks["size"].resample("1min").sum()
bars["vwap"] = ticks["pv"].resample("1min").sum() / bars["volume"]
bars = bars.dropna()
bars.to_parquet("es_1min.parquet")
print(bars.tail())

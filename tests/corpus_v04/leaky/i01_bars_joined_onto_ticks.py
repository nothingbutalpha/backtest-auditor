# expect: LA12:medium
import pandas as pd

ticks = pd.read_parquet("es_ticks.parquet")  # columns: price, size; DatetimeIndex
ticks["pv"] = ticks["price"] * ticks["size"]
bars = ticks["price"].resample("1min").ohlc()
bars["vwap"] = ticks["pv"].resample("1min").sum() / ticks["size"].resample("1min").sum()
ticks["bar_vwap"] = bars["vwap"].reindex(ticks.index, method="ffill")
ticks["signal"] = (ticks["price"] < ticks["bar_vwap"]).astype(int)
print(ticks["signal"].mean())

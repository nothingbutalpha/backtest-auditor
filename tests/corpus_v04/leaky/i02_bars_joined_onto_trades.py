# expect: LA12:medium
import pandas as pd

ticks = pd.read_parquet("es_ticks.parquet")  # columns: price, size; DatetimeIndex
ticks["pv"] = ticks["price"] * ticks["size"]
bars = ticks["price"].resample("1min").ohlc()
bars["volume"] = ticks["size"].resample("1min").sum()
fills = pd.read_csv("my_fills.csv", index_col="time", parse_dates=True)
fills = fills.join(bars, how="left")
fills["slippage"] = fills["fill_price"] / fills["close"] - 1
print(fills["slippage"].mean())

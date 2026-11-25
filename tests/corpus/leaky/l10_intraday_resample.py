# expect: LA12:medium
import pandas as pd

ticks = pd.read_csv("trades.csv", parse_dates=["timestamp"], index_col="timestamp")
bars = ticks["price"].resample("5min").ohlc()
quotes = pd.read_csv("quotes.csv", parse_dates=["timestamp"], index_col="timestamp")
merged = bars.join(quotes["mid"], how="left")

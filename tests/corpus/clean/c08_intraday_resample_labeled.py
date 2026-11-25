# expect: none
import pandas as pd

ticks = pd.read_csv("trades.csv", parse_dates=["timestamp"], index_col="timestamp")
bars = ticks["price"].resample("5min", label="right", closed="left").ohlc()
daily = ticks["price"].resample("D").last()
weekly = daily.resample("W-FRI").last()

import pandas as pd

quotes = pd.read_csv("quotes.csv", parse_dates=["ts"], index_col="ts")
quotes["mid"] = (quotes["bid"] + quotes["ask"]) / 2
quotes["mid"] = quotes["mid"].ffill(limit=5)
quotes["spread"] = (quotes["ask"] - quotes["bid"]).fillna(0)
daily = quotes["mid"].resample("1D").last().ffill()
daily.to_csv("daily_mid.csv")

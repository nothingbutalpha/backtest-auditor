import pandas as pd

px = pd.read_csv("ohlc.csv", index_col=0, parse_dates=True).sort_index()
px[["open", "high", "low", "close"]] = px[["open", "high", "low", "close"]].bfill(axis=1)
px["signal"] = (px["close"] > px["close"].rolling(20).mean()).astype(int)

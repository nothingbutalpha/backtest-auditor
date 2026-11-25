import pandas as pd

prices = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date").sort_index()
prices["ret"] = prices["close"].pct_change()
prices["mom20"] = prices["close"].pct_change(-20)
prices["trend"] = prices["close"].rolling(50, center=True).mean()
prices["raw"] = ((prices["mom20"] > 0) & (prices["close"] > prices["trend"])).astype(int)
prices["signal"] = prices["raw"].shift(-1)
prices.to_csv("features.csv")

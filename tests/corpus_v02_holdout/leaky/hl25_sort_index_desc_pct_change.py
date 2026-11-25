# expect: LA02
# Index sorted descending, then pct_change/rolling features computed in reverse time order.
import numpy as np
import pandas as pd

prices = pd.read_csv("copper.csv", index_col="date", parse_dates=True).sort_index(ascending=False)
prices["mom10"] = prices["close"].pct_change(10)
prices["vol20"] = prices["close"].pct_change().rolling(20).std()
prices["score"] = prices["mom10"] / prices["vol20"]

prices = prices.sort_index()
prices["ret"] = prices["close"].pct_change()
prices["pos"] = np.sign(prices["score"]).shift(1).fillna(0)
prices["net"] = prices["pos"] * prices["ret"] - 0.0007 * prices["pos"].diff().abs()
print("Sharpe", prices["net"].mean() / prices["net"].std() * np.sqrt(252))

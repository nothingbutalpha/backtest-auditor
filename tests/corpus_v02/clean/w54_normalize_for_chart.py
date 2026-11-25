# expect: none
import pandas as pd
import matplotlib.pyplot as plt
prices = pd.read_csv("closes.csv", index_col=0, parse_dates=True)
rebased = prices / prices.iloc[0] * 100
scaled = prices["SPY"] / prices["SPY"].max()
rebased.plot()
scaled.plot()
plt.show()

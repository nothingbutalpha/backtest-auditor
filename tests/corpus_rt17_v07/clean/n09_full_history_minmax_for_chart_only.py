# expect: none
import pandas as pd
import matplotlib.pyplot as plt
px = pd.read_csv("ko.csv", index_col=0, parse_dates=True)["close"]
def rescale(s):
    return (s - s.min()) / (s.max() - s.min())
sig = (px > px.rolling(100).mean()).astype(float).shift(1)
pnl = sig * px.pct_change()
plt.plot(rescale(px), label="price (rescaled)")
plt.plot(rescale(pnl.cumsum()), label="strategy (rescaled)")
plt.legend()
plt.savefig("chart.png")

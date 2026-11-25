# expect: none
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
df["mom"] = df["close"].pct_change(20)
norm = (df["close"] - df["close"].min()) / (df["close"].max() - df["close"].min())
plt.plot(norm)
plt.show()

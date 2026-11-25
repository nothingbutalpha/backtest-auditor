# expect: none
"""How often does the price rise 5% within 20 bars? A base rate for the label, nothing more."""
import numpy as np
import pandas as pd

c = pd.read_csv("btc.csv")["close"].to_numpy()
target = []
for i in range(len(c) - 21):
    if c[i + 1: i + 21].max() >= c[i] * 1.05:
        target.append(1)
    else:
        target.append(0)
print("base rate", np.mean(target))

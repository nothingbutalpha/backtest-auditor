# expect: none
import numpy as np
import pandas as pd

def growth_rate(px):
    x = np.arange(len(px))
    slope, _ = np.polyfit(x, np.log(px.values), 1)
    return slope * 252

copper = pd.read_csv("hg.csv", index_col=0, parse_dates=True)["close"]
print("log-linear growth per year:", growth_rate(copper))

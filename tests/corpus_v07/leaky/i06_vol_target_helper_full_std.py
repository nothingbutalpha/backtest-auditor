# expect: LA09|LA05@6
import numpy as np
import pandas as pd

def lever_to(r, target=0.12):
    ann_vol = r.std() * np.sqrt(252)
    return r * (target / ann_vol)

daily = pd.read_csv("trend_book.csv", index_col=0, parse_dates=True)["ret"]
levered = lever_to(daily)
equity = (1 + levered).cumprod()
print("CAGR", equity.iloc[-1] ** (252 / len(equity)) - 1)

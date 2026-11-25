# expect: none
import numpy as np
import pandas as pd

def lever_to(r, target=0.12, window=60):
    ann_vol = r.rolling(window).std().shift(1) * np.sqrt(252)
    return r * (target / ann_vol)

daily = pd.read_csv("trend_book.csv", index_col=0, parse_dates=True)["ret"]
levered = lever_to(daily).dropna()
equity = (1 + levered).cumprod()
print("CAGR", equity.iloc[-1] ** (252 / len(equity)) - 1)

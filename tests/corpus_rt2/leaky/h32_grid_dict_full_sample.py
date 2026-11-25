import pandas as pd, numpy as np
df = pd.read_csv("spy.csv", index_col=0, parse_dates=True)
ret = df["Close"].pct_change()
results = {}
for fast in [5, 10, 20, 50]:
    for slow in [100, 150, 200]:
        sig = (df["Close"].rolling(fast).mean() > df["Close"].rolling(slow).mean()).astype(int).shift(1)
        s = sig * ret
        results[(fast, slow)] = s.mean() / s.std() * np.sqrt(252)
best = max(results, key=results.get)
